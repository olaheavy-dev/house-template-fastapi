"""Streaming through the redacting chat model (#78): placeholders split across chunks, JSON text,
one reservation per call, and what a consumer that stops or is cancelled leaves behind."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall
from app.llm.core.redaction import Redactor, redaction_scope
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel
from tests import wire
from tests.wire import EMAIL, NAME, PHONE, SEEDS, STREAMING_PROVIDERS, TRICKY

pytestmark = pytest.mark.anyio


def stream_reply(provider: str, pieces: list[str]) -> wire.Reply:
    if provider == "openai":
        data = wire.openai_stream_text(pieces)
    elif provider == "openai-responses":
        data = wire.responses_stream_text(pieces)
    else:
        data = wire.anthropic_stream_text(pieces)
    return lambda _body: data


def token_text(token: Any) -> str:
    """The text of a streamed token: a string, or a list of content blocks."""
    if isinstance(token, str):
        return token
    return "".join(b.get("text", "") for b in token if isinstance(b, dict))


def merged(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
async def test_a_placeholder_split_across_chunks_is_restored_and_no_half_one_is_emitted(
    provider: str,
) -> None:
    pieces = ["Write to [EM", "AIL_", "1] and call [PHO", "NE_1", "] today"]
    guard = wire.Recorder()
    inner = wire.raw_model(provider, callbacks=[guard])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    outer = wire.Recorder()
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(stream_reply(provider, pieces)) as sent:
            chunks = [
                chunk
                async for chunk in model.astream(
                    [HumanMessage(f"{NAME} {EMAIL} {PHONE}")], config={"callbacks": [outer]}
                )
            ]
    for seed in SEEDS:
        assert seed not in sent.text
    assert "".join(chunk.text for chunk in chunks) == f"Write to {EMAIL} and call {PHONE} today"
    # A held-back "[EM" is never shown: no chunk has an unclosed bracket.
    assert not any("[" in chunk.text for chunk in chunks)
    # The budget side (the inner run) saw exactly one start and one end: one reservation.
    assert guard.names() == ["start", "end"]
    # What a stream consumer is told comes from one run (the wrapper's), restored.
    assert len({run for run, _ in outer.tokens}) == 1
    assert "[" not in "".join(token_text(token) for _, token in outer.tokens)
    # Usage rides the merged message, as it does for the raw model.
    usage = merged(chunks).usage_metadata
    assert usage is not None
    assert usage["output_tokens"] == 7


@pytest.mark.parametrize("provider", STREAMING_PROVIDERS)
async def test_a_stream_with_nothing_to_redact_matches_the_raw_model(provider: str) -> None:
    pieces = ["A freehold ", "is ownership ", "of the land."]
    inner, model = wire.wrapped(provider)
    with wire.captured(stream_reply(provider, pieces)) as raw_sent:
        raw = [chunk.text async for chunk in inner.astream([HumanMessage("What is a freehold?")])]
    with wire.captured(stream_reply(provider, pieces)) as wrapped_sent:
        wrapped = [
            chunk.text async for chunk in model.astream([HumanMessage("What is a freehold?")])
        ]
    assert wrapped_sent.bodies == raw_sent.bodies
    assert "".join(wrapped) == "".join(raw) == "A freehold is ownership of the land."


async def test_text_still_held_back_when_the_stream_ends_is_released() -> None:
    # "[EMA" could have become a placeholder, but the stream ended: it is text, and it is shown.
    _, model = wire.wrapped("openai")
    with wire.captured(stream_reply("openai", ["See you [EMA"])):
        chunks = [chunk async for chunk in model.astream([HumanMessage("hi")])]
    assert "".join(chunk.text for chunk in chunks) == "See you [EMA"


async def test_streamed_placeholders_the_model_invented_stay_and_are_counted() -> None:
    _, model = wire.wrapped("openai")
    with capture_logs() as logs:
        async with redaction_scope(known={"NAME": [NAME]}):
            with wire.captured(stream_reply("openai", ["Hi [NAME_", "1] and [EMAIL_", "9]"])):
                chunks = [chunk async for chunk in model.astream([HumanMessage(f"I am {NAME}")])]
    assert "".join(chunk.text for chunk in chunks) == f"Hi {NAME} and [EMAIL_9]"
    [event] = [e for e in logs if e["event"] == "llm_unrestored_placeholders"]
    assert event["count"] == 1
    assert NAME not in repr(logs)


@pytest.mark.parametrize("provider", ["openai", "openai-responses", "anthropic"])
async def test_json_mode_text_streams_restored_with_quotes_and_backslashes_escaped(
    provider: str,
) -> None:
    # The model writes a JSON document whose "name" is a placeholder; the original holds a quote
    # and a backslash, so it must be put back JSON-escaped or the document is not JSON.
    document = '{"name": "[NAME_1]", "email": "[EMAIL_1]"}'
    pieces = [document[:12], document[12:20], document[20:31], document[31:]]
    _, model = wire.wrapped(provider)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        "required": ["name", "email"],
        "additionalProperties": False,
    }
    # Anthropic has no "json_object" mode: its JSON answers come from a schema.
    response_format: dict[str, Any] = (
        {"type": "json_schema", "json_schema": {"name": "contact", "schema": schema}}
        if provider == "anthropic"
        else {"type": "json_object"}
    )
    bound = model.bind(response_format=response_format)
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(stream_reply(provider, pieces)) as sent:
            chunks = [
                chunk async for chunk in bound.astream([HumanMessage(f"{TRICKY} at {EMAIL}")])
            ]
    assert TRICKY not in sent.text
    assert json.loads("".join(chunk.text for chunk in chunks)) == {"name": TRICKY, "email": EMAIL}


# --- stopping early ------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_a_consumer_that_stops_early_leaves_the_same_run_events_as_for_the_raw_model(
    provider: str,
) -> None:
    pieces = ["one ", "two ", "three ", "four"]
    raw_events = wire.Recorder()
    raw = wire.raw_model(provider, callbacks=[raw_events])
    with wire.captured(stream_reply(provider, pieces)):
        stream = cast(AsyncGenerator[Any, None], raw.astream([HumanMessage("hi")]))
        async for _ in stream:
            break
        await stream.aclose()

    inner_events = wire.Recorder()
    inner = wire.raw_model(provider, callbacks=[inner_events])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    with wire.captured(stream_reply(provider, pieces)):
        stream = cast(AsyncGenerator[Any, None], model.astream([HumanMessage("hi")]))
        async for _ in stream:
            break
        await stream.aclose()
    # The inner run (where the budget guard sits) starts and then ends, as without the wrapper: it
    # is closed with the stream, not left open with its reservation leaking.
    assert inner_events.names() == raw_events.names()
    assert inner_events.names()[0] == "start"
    assert len(inner_events.names()) == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_a_cancelled_task_leaves_the_same_run_events_as_for_the_raw_model(
    provider: str,
) -> None:
    pieces = ["one ", "two ", "three ", "four"]

    async def consume(model: Any, started: asyncio.Event) -> None:
        async for _ in model.astream([HumanMessage("hi")]):
            started.set()
            await asyncio.sleep(10)  # a slow consumer, to be cancelled here

    async def run(model: Any) -> None:
        started = asyncio.Event()
        with wire.captured(stream_reply(provider, pieces)):
            task = asyncio.create_task(consume(model, started))
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # The abandoned streams are closed by the event loop's finalizer, a few turns later.
        await asyncio.sleep(0.2)

    raw_events = wire.Recorder()
    await run(wire.raw_model(provider, callbacks=[raw_events]))
    inner_events = wire.Recorder()
    inner = wire.raw_model(provider, callbacks=[inner_events])
    await run(RedactingChatModel(inner, Redactor(wire.DETECTORS)))
    assert inner_events.names() == raw_events.names()
    assert inner_events.names()[0] == "start"


async def test_a_cancelled_stream_leaves_the_same_ledger_row_as_the_raw_model(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    # Against the real ledger: whatever a cancel leaves for the provider model alone (a settled,
    # failed or still reserved row), it leaves for the wrapped one. The hidden inner run is closed
    # with the stream, so its reservation is not left open any longer than without the wrapper.
    gateway: Gateway = app.state.gateway
    assert gateway.redactor is not None
    pieces = ["one ", "two ", "three ", "four"]

    async def cancel_mid_stream(model: Any) -> None:
        started = asyncio.Event()

        async def consume() -> None:
            async for _ in model.astream([HumanMessage("hi")]):
                started.set()
                await asyncio.sleep(10)

        with wire.captured(stream_reply("openai", pieces)):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.sleep(0.3)  # the abandoned streams are closed by the loop's finalizer

    await cancel_mid_stream(wire.raw_model("openai", callbacks=[gateway.guard]))
    first = {row.id: row.status for row in await db_session.scalars(select(LlmCall))}
    assert len(first) == 1
    inner = wire.raw_model("openai", callbacks=[gateway.guard])
    await cancel_mid_stream(RedactingChatModel(inner, gateway.redactor))
    second = {
        row.id: row.status
        for row in await db_session.scalars(select(LlmCall))
        if row.id not in first
    }
    assert list(second.values()) == list(first.values())


@pytest.mark.parametrize("provider", ["openai", "openai-responses", "anthropic"])
async def test_a_json_answer_is_restored_the_same_way_whole_and_streamed(provider: str) -> None:
    # One rule for both paths: no response_format is bound (the prompt asked for JSON), and the
    # answer opens with "{", so the original, which holds a quote and a backslash, is escaped.
    document = '{"name": "[NAME_1]", "email": "[EMAIL_1]"}'
    pieces = [document[:3], document[3:14], document[14:27], document[27:]]
    _, model = wire.wrapped(provider)
    messages = [HumanMessage(f"Reply in JSON for {TRICKY} at {EMAIL}")]
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.text_reply(provider, document)):
            whole = (await model.ainvoke(messages)).text
        with wire.captured(stream_reply(provider, pieces)):
            streamed = "".join([c.text async for c in model.astream(messages)])
    expected = {"name": TRICKY, "email": EMAIL}
    assert json.loads(whole) == expected
    assert json.loads(streamed) == expected


async def test_anthropic_thinking_deltas_pass_unchanged_and_do_not_disturb_the_text() -> None:
    # Thinking is produced from placeholders and carries a signature the provider checks, so it is
    # neither restored nor changed; the text block after it is restored as usual.
    thinking = [
        {"type": "thinking_delta", "thinking": "Mention [NAME_1] and "},
        {"type": "thinking_delta", "thinking": "[EMAIL_1]."},
        {"type": "signature_delta", "signature": "sig-xyz"},
    ]
    text = [{"type": "text_delta", "text": t} for t in ("Hi [NA", "ME_1], mail [EMAIL", "_1].")]
    data = wire.anthropic_stream(
        [
            ({"type": "thinking", "thinking": "", "signature": ""}, thinking),
            ({"type": "text", "text": ""}, text),
        ]
    )
    _, model = wire.wrapped("anthropic")
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(lambda _b: data):
            chunks = [c async for c in model.astream([HumanMessage(f"I am {NAME}, {EMAIL}")])]
    total = merged(chunks)
    assert isinstance(total.content, list)
    [block] = [b for b in total.content if isinstance(b, dict)]
    assert block["thinking"] == "Mention [NAME_1] and [EMAIL_1]."
    assert block["signature"] == "sig-xyz"
    assert total.text == f"Hi {NAME}, mail {EMAIL}."


ADDRESS = '12 "Quoted" Row\\Flat\nLeeds'


@pytest.mark.parametrize("provider", ["openai", "openai-responses", "anthropic"])
async def test_a_plain_answer_that_starts_with_a_placeholder_is_restored_exactly(
    provider: str,
) -> None:
    # Every placeholder starts with "[", so an answer opening with one is not JSON: the originals,
    # which hold a quote, a backslash and a newline, go back as they were, whole and streamed.
    answer = "[NAME_1], here is your draft. Address: [ADDRESS_1]."
    pieces = ["[", "NAME_1], here is ", "your draft. Address: [ADDR", "ESS_1]."]
    expected = f"{TRICKY}, here is your draft. Address: {ADDRESS}."
    _, model = wire.wrapped(provider)
    messages = [HumanMessage(f"I am {TRICKY} at {ADDRESS}")]
    async with redaction_scope(known={"NAME": [TRICKY], "ADDRESS": [ADDRESS]}):
        with wire.captured(lambda _b: wire.text_reply(provider, answer)):
            whole = (await model.ainvoke(messages)).text
        with wire.captured(stream_reply(provider, pieces)):
            streamed = "".join([c.text async for c in model.astream(messages)])
    assert whole == expected
    assert streamed == expected


async def test_a_lone_opening_bracket_at_the_end_of_a_stream_is_still_shown() -> None:
    _, model = wire.wrapped("openai")
    with wire.captured(stream_reply("openai", ["See ", "["])):
        chunks = [c async for c in model.astream([HumanMessage("hi")])]
    assert "".join(c.text for c in chunks) == "See ["
    with wire.captured(stream_reply("openai", ["["])):
        chunks = [c async for c in model.astream([HumanMessage("hi")])]
    assert "".join(c.text for c in chunks) == "["


@pytest.mark.parametrize("opening", ['{"a": ', '["x", ', "[1, ", "[ 2"])
async def test_text_that_opens_like_a_json_document_is_taken_as_json(opening: str) -> None:
    from app.llm.langchain.redaction import _opens_json

    assert _opens_json(opening)


@pytest.mark.parametrize(
    "opening", ["[NAME_1], hi", "  [EMAIL_2]", "[", "Hello {", "[a", "[ NAME_1]"]
)
async def test_text_that_opens_with_a_placeholder_or_plain_words_is_not(opening: str) -> None:
    from app.llm.langchain.redaction import _opens_json

    assert not _opens_json(opening)
