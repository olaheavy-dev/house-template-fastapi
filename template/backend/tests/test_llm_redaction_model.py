"""The redacting chat model (#78): what a plain `await model.ainvoke(...)` sends and returns.

The real OpenAI (Chat Completions and Responses) and Anthropic SDK models are used, with the HTTP
layer replaced (tests/wire.py), so a test reads the bytes that would leave the machine. Every
personal value is synthetic. Expected placeholders are worked out by hand from the order of
first appearance: the system text comes first (name, then email), then the human text (phone,
then NINO).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from structlog.testing import capture_logs

from app.llm import prices
from app.llm.core.redaction import RedactionFailed, Redactor, redaction_scope
from app.llm.langchain import redaction
from app.llm.langchain.callbacks import BudgetGuard
from app.llm.langchain.redaction import RedactingChatModel
from tests import wire
from tests.wire import EMAIL, HISTORY, NAME, NINO, PHONE, PROVIDERS, SEEDS

pytestmark = pytest.mark.anyio

PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_ainvoke_sends_no_personal_value_and_restores_them_in_the_answer(
    provider: str,
) -> None:
    _, model = wire.wrapped(provider)
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(wire.echo(provider)) as sent:
            answer = await model.ainvoke(HISTORY)
    assert len(sent.bodies) == 1
    for seed in SEEDS:
        assert seed not in sent.text
    # Numbered by first appearance across the whole message list, repeats included.
    for placeholder in ("[NAME_1]", "[EMAIL_1]", "[PHONE_1]", "[NINO_1]"):
        assert placeholder in sent.text
    assert "[NAME_2]" not in sent.text
    assert "[EMAIL_2]" not in sent.text
    # The model echoed the placeholders it saw; the caller gets the originals back.
    for seed in SEEDS:
        assert seed in answer.text
    assert PLACEHOLDER.search(answer.text) is None


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_handler_passed_by_the_caller_sees_only_the_public_run(provider: str) -> None:
    # The hidden inner run inherits no handler of the caller's (LangGraph's message stream is one),
    # so the caller is told about one model run, with its own text, and nothing twice.
    _, model = wire.wrapped(provider)
    outer = wire.Recorder()
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(wire.echo(provider)):
            await model.ainvoke(HISTORY, config={"callbacks": [outer]})
    assert outer.names() == ["start", "end"]
    [(_, seen, _)] = outer.starts
    assert EMAIL in repr(seen)  # the caller's own run shows the caller's own text
    assert PLACEHOLDER.search(repr(seen)) is None


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_request_with_nothing_to_redact_is_byte_identical_to_the_raw_models(
    provider: str,
) -> None:
    inner, model = wire.wrapped(provider)
    messages: list[BaseMessage] = [SystemMessage("Be brief."), HumanMessage("What is a freehold?")]
    with wire.captured(lambda _body: wire.text_reply(provider, "ok")) as raw:
        await inner.ainvoke(messages)
    with wire.captured(lambda _body: wire.text_reply(provider, "ok")) as wrapped:
        answer = await model.ainvoke(messages)
    assert wrapped.bodies == raw.bodies
    assert answer.text == "ok"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_text_blocks_and_tool_results_in_list_content_are_redacted(provider: str) -> None:
    _, model = wire.wrapped(provider)
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": f"Mail {EMAIL}"},
                {"type": "text", "text": f"Call {PHONE}"},
            ]
        ),
        AIMessage(
            "",
            tool_calls=[{"name": "lookup", "args": {"note": {"nested": [NINO]}}, "id": "call_1"}],
        ),
        # A tool result whose content is a list of blocks.
        ToolMessage(content=[{"type": "text", "text": f"{NAME} is {NINO}"}], tool_call_id="call_1"),
    ]
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(wire.echo(provider)) as sent:
            answer = await model.ainvoke(messages)
    for seed in (EMAIL, PHONE, NINO, NAME):
        assert seed not in sent.text
    for seed in (EMAIL, PHONE, NINO, NAME):
        assert seed in answer.text


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_an_image_block_is_sent_untouched_and_the_text_beside_it_is_redacted(
    provider: str,
) -> None:
    inner, model = wire.wrapped(provider)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    messages: list[BaseMessage] = [
        HumanMessage(content=[{"type": "text", "text": f"Who is {EMAIL}?"}, image])
    ]
    with wire.captured(wire.echo(provider)) as sent:
        await model.ainvoke(messages)
    assert EMAIL not in sent.text
    assert "iVBORw0KGgo=" in sent.text  # the image is sent as it was
    # Same bytes as the raw model for the same already-redacted text.
    redacted: list[BaseMessage] = [
        HumanMessage(content=[{"type": "text", "text": "Who is [EMAIL_1]?"}, image])
    ]
    with wire.captured(wire.echo(provider)) as expected:
        await inner.ainvoke(redacted)
    assert sent.bodies == expected.bodies


async def test_anthropic_thinking_blocks_are_neither_redacted_nor_restored() -> None:
    # A provider checks a thinking block's signature, so changing its text breaks the next call.
    # The model thought about placeholders, so the block keeps them.
    _, model = wire.wrapped("anthropic")
    thinking = {"type": "thinking", "thinking": "I will mention [NAME_1].", "signature": "sig-abc"}
    messages: list[BaseMessage] = [
        HumanMessage(f"Hello, I am {NAME}"),
        AIMessage(content=[thinking, {"type": "text", "text": f"Hi {NAME}"}]),
        HumanMessage("Summarise."),
    ]
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(
            lambda _b: wire.anthropic_text("Bye [NAME_1]", thinking="Think of [NAME_1].")
        ) as sent:
            answer = await model.ainvoke(messages)
    body = sent.json_bodies()[0]
    blocks: list[dict[str, Any]] = [
        b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]
    ]
    [sent_thinking] = [b for b in blocks if b["type"] == "thinking"]
    assert sent_thinking == {**thinking}
    assert NAME not in sent.text
    assert isinstance(answer.content, list)
    returned_thinking = next(
        b for b in answer.content if isinstance(b, dict) and b["type"] == "thinking"
    )
    assert returned_thinking["thinking"] == "Think of [NAME_1]."
    assert returned_thinking["signature"] == "sig-abc"
    assert answer.text == f"Bye {NAME}"


async def test_a_placeholder_the_model_invents_stays_and_is_logged_as_a_count() -> None:
    _, model = wire.wrapped("openai")
    with (
        capture_logs() as logs,
        wire.captured(lambda _b: wire.openai_text("Hi [NAME_1] and [EMAIL_9] and [PHONE_77]")),
    ):
        async with redaction_scope(known={"NAME": [NAME]}):
            answer = await model.ainvoke([HumanMessage(f"I am {NAME}")])
    assert answer.text == f"Hi {NAME} and [EMAIL_9] and [PHONE_77]"
    [event] = [e for e in logs if e["event"] == "llm_unrestored_placeholders"]
    assert event["count"] == 2
    assert NAME not in repr(logs)


# --- the sync API, and what it looks like to callers ---------------------------------------------


def test_the_sync_api_is_refused() -> None:
    _, model = wire.wrapped("openai")
    with pytest.raises(NotImplementedError, match="async"):
        model.invoke([HumanMessage("hi")])


def test_what_create_agent_and_the_step_limit_read_comes_from_the_provider_model() -> None:
    for provider in PROVIDERS:
        inner, model = wire.wrapped(provider)
        assert model._llm_type == inner._llm_type
        assert model.profile == inner.profile
        assert getattr(model, "model_name", None) == getattr(inner, "model_name", None)
        assert model._get_ls_params() == inner._get_ls_params()
        assert model._identifying_params == inner._identifying_params


def test_the_only_visible_difference_from_the_provider_model_is_its_type() -> None:
    # The breaking change (#78): code that checks isinstance(gateway.model(), ChatOpenAI) fails.
    openai_inner, openai_model = wire.wrapped("openai")
    anthropic_inner, anthropic_model = wire.wrapped("anthropic")
    assert isinstance(openai_inner, ChatOpenAI)
    assert not isinstance(openai_model, ChatOpenAI)
    assert isinstance(anthropic_inner, ChatAnthropic)
    assert not isinstance(anthropic_model, ChatAnthropic)
    assert isinstance(openai_model, RedactingChatModel)
    assert openai_model.inner is openai_inner


# --- failing closed ------------------------------------------------------------------------------


class Boom:
    kind = "EMAIL"

    def find(self, text: str) -> Any:
        raise RuntimeError(f"cannot read {text}")  # an error whose message holds the text


class StubMeter:
    """What the guard calls when it reserves: records what it was asked."""

    def __init__(self) -> None:
        self.reserved: list[dict[str, Any]] = []
        self.settled = 0

    async def reserve(self, _run_id: Any, **given: Any) -> int:
        self.reserved.append(given)
        return 0

    async def settle(self, *_a: Any, **_k: Any) -> None:
        self.settled += 1

    async def failed(self, *_a: Any, **_k: Any) -> None:
        pass


def stub_guard() -> tuple[BudgetGuard, StubMeter]:
    name = "openai:gpt-6-luna"
    guard = BudgetGuard(
        model=name,
        price=prices.PRICES[name],
        cap_micro_usd=10**9,
        max_tokens=256,
        sessionmaker=lambda: None,  # type: ignore[arg-type,return-value]
    )
    meter = StubMeter()
    guard.meter = meter  # type: ignore[assignment]
    return guard, meter


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_detector_that_raises_sends_nothing_reserves_nothing_and_leaks_nothing(
    provider: str,
) -> None:
    guard, meter = stub_guard()
    inner = wire.raw_model(provider, callbacks=[guard])
    model = RedactingChatModel(inner, Redactor([Boom()]))
    secret = "top-secret-text-7788"  # noqa: S105 - test data
    with (
        capture_logs() as logs,
        wire.captured(lambda _b: wire.text_reply(provider, "x")) as sent,
        pytest.raises(RedactionFailed) as caught,
    ):
        await model.ainvoke([HumanMessage(secret)])
    assert sent.bodies == []  # zero HTTP requests
    assert meter.reserved == []  # so no llm_call row either
    assert caught.value.status == 503
    assert secret not in repr(caught.value)
    assert secret not in str(caught.value.detail)
    assert caught.value.__cause__ is None
    [event] = [e for e in logs if e["event"] == "llm_redaction_failed"]
    assert event["error_type"] == "RuntimeError"
    assert secret not in repr(logs)


async def test_any_error_while_preparing_the_request_fails_closed_and_says_only_its_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("odd text that must not be logged")

    monkeypatch.setattr(redaction, "_walk_message", broken)
    _, model = wire.wrapped("openai")
    with (
        capture_logs() as logs,
        wire.captured(lambda _b: wire.openai_text("ok")) as sent,
        pytest.raises(RedactionFailed) as caught,
    ):
        await model.ainvoke([HumanMessage("hello")])
    assert sent.bodies == []
    assert caught.value.__cause__ is None
    [event] = [e for e in logs if e["event"] == "llm_redaction_failed"]
    assert event["error_type"] == "ValueError"
    assert "odd text" not in repr(logs)


# --- the budget sees redacted text ---------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_the_budget_guard_sees_redacted_messages_and_reserves_for_them(
    provider: str,
) -> None:
    guard, meter = stub_guard()
    recorder = wire.Recorder()
    inner = wire.raw_model(provider, callbacks=[guard, recorder])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    async with redaction_scope(known={"NAME": [NAME]}):
        with wire.captured(wire.echo(provider)):
            await model.ainvoke(HISTORY)
    [(_, seen, metadata)] = recorder.starts
    for seed in SEEDS:
        assert seed not in repr(seen)
    # System: name, email. Human: phone, NINO. AI text: name. Its args: name, email. Tool result:
    # name, phone. That is 2 + 2 + 1 + 2 + 2 = 9 replaced values (occurrences, not distinct ones).
    assert metadata["redacted_count"] == 9
    assert len(meter.reserved) == 1
    assert meter.settled == 1

    # Test 28: the reservation is what a raw model reserves for the redacted messages themselves,
    # so it measures exactly what is sent (a longer original would reserve more).
    guard2, meter2 = stub_guard()
    reference = wire.raw_model(provider, callbacks=[guard2])
    with wire.captured(wire.echo(provider)):
        await reference.ainvoke(seen[0])
    assert meter.reserved[0]["input_tokens"] == meter2.reserved[0]["input_tokens"]
    # ...and the redacted request is shorter than the original one, which would reserve more.
    guard3, meter3 = stub_guard()
    original = wire.raw_model(provider, callbacks=[guard3])
    with wire.captured(wire.echo(provider)):
        await original.ainvoke(HISTORY)
    assert meter.reserved[0]["input_tokens"] != meter3.reserved[0]["input_tokens"]


# --- concurrency: nothing is shared between calls ------------------------------------------------


async def test_concurrent_calls_keep_their_own_mappings_scopes_and_counts() -> None:
    guard, meter = stub_guard()
    recorder = wire.Recorder()
    inner = wire.raw_model("openai", callbacks=[guard, recorder])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))

    async def ask(email: str, name: str, extra: str) -> str:
        async with redaction_scope(known={"NAME": [name]}):
            return (await model.ainvoke([HumanMessage(f"{name} {email} {extra}")])).text

    with wire.captured(wire.echo("openai")) as sent:
        answers = await asyncio.gather(
            ask("a.one@example.com", "Ann One", ""),
            ask("b.two@example.com", "Bob Two", PHONE),
            ask("c.three@example.com", "Cat Three", f"{PHONE} {NINO}"),
        )
    assert len(sent.bodies) == 3
    for body in sent.text.splitlines():
        assert "example.com" not in body
        assert "One" not in body
        assert "Two" not in body
    # Each answer holds its own caller's values (the same placeholder numbers meant different
    # values in the three calls).
    assert "Ann One" in answers[0]
    assert "a.one@example.com" in answers[0]
    assert "Bob Two" in answers[1]
    assert "b.two@example.com" in answers[1]
    assert "Cat Three" in answers[2]
    assert "c.three@example.com" in answers[2]
    assert "Bob Two" not in answers[0]
    assert "Ann One" not in answers[1]
    # Counts travel with their own call: 2, 3 and 4 values.
    assert sorted(m["redacted_count"] for _, _, m in recorder.starts) == [2, 3, 4]
    assert len(meter.reserved) == 3
