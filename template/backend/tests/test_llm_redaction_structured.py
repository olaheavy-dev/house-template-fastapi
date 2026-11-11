"""Structured output through the redacting chat model (#78, test 29): every method the OpenAI and
Anthropic integrations offer, with a pydantic and a dict schema, with and without `include_raw`.
The request holds placeholders only, and the parsed fields come back as the originals, nested ones
included, even when an original holds a quote or a backslash."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, EmailStr

from app.llm.core.redaction import redaction_scope
from app.llm.langchain.redaction import RedactionUnavailable
from tests import wire
from tests.wire import DICT_SCHEMA, EMAIL, TRICKY, Contact

pytestmark = pytest.mark.anyio

ADDRESS = '12 "Quoted" Row\\Flat'
PLACEHOLDERS = {
    "name": "[NAME_1]",
    "email": "[EMAIL_1]",
    "address": {"city": "Leeds", "line": "[ADDRESS_1]"},
}
EXPECTED = {"name": TRICKY, "email": EMAIL, "address": {"city": "Leeds", "line": ADDRESS}}

# (provider, method, extra arguments): each pairing the integrations support.
METHODS: list[tuple[str, str, dict[str, Any]]] = [
    ("openai", "function_calling", {}),
    ("openai", "json_mode", {}),
    ("openai", "json_schema", {"strict": True}),
    ("openai", "json_schema", {"strict": False}),
    ("openai-responses", "function_calling", {}),
    ("openai-responses", "json_schema", {"strict": True}),
    ("anthropic", "function_calling", {}),
    ("anthropic", "json_schema", {}),
]


def structured_reply(provider: str, method: str, payload: dict[str, Any]) -> wire.Reply:
    if method == "function_calling":
        return lambda _b: wire.tool_reply(provider, "Contact", payload)
    return lambda _b: wire.text_reply(provider, json.dumps(payload))


def as_dict(parsed: Any) -> dict[str, Any]:
    return parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)


@pytest.mark.parametrize("schema_kind", ["pydantic", "dict"])
@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize(("provider", "method", "extra"), METHODS)
async def test_structured_output_sends_placeholders_and_returns_the_originals(
    provider: str, method: str, extra: dict[str, Any], include_raw: bool, schema_kind: str
) -> None:
    _, model = wire.wrapped(provider)
    schema: Any = Contact if schema_kind == "pydantic" else DICT_SCHEMA
    runnable = model.with_structured_output(schema, method=method, include_raw=include_raw, **extra)
    async with redaction_scope(known={"NAME": [TRICKY], "ADDRESS": [ADDRESS]}):
        with wire.captured(structured_reply(provider, method, PLACEHOLDERS)) as sent:
            out = await runnable.ainvoke([HumanMessage(f"{TRICKY} {EMAIL} lives at {ADDRESS}")])
    for seed in (EMAIL, TRICKY, ADDRESS):
        assert seed not in sent.text
    assert len(sent.bodies) == 1
    if include_raw:
        assert out["parsing_error"] is None
        parsed = out["parsed"]
        assert "[NAME_1]" not in repr(out["raw"].content)
        assert "[EMAIL_1]" not in repr(out["raw"].tool_calls)
    else:
        parsed = out
    assert as_dict(parsed) == EXPECTED


@pytest.mark.parametrize(("provider", "method", "extra"), METHODS)
async def test_a_structured_request_with_nothing_to_redact_matches_the_raw_models(
    provider: str, method: str, extra: dict[str, Any]
) -> None:
    inner, model = wire.wrapped(provider)
    payload = {"name": "x", "email": "y", "address": {"city": "Leeds", "line": "z"}}
    messages = [SystemMessage("Extract."), HumanMessage("Leeds flat")]
    reply = structured_reply(provider, method, payload)
    with wire.captured(reply) as raw:
        await inner.with_structured_output(
            Contact, method=method, include_raw=True, **extra
        ).ainvoke(messages)
    with wire.captured(reply) as wrapped:
        await model.with_structured_output(
            Contact, method=method, include_raw=True, **extra
        ).ainvoke(messages)
    assert wrapped.bodies == raw.bodies


async def test_the_raw_message_of_a_structured_answer_holds_restored_text_too() -> None:
    # OpenAI's json_schema answer is JSON text: restored JSON-escaped, so it still parses.
    _, model = wire.wrapped("openai")
    runnable = model.with_structured_output(
        Contact, method="json_schema", include_raw=True, strict=True
    )
    async with redaction_scope(known={"NAME": [TRICKY], "ADDRESS": [ADDRESS]}):
        with wire.captured(structured_reply("openai", "json_schema", PLACEHOLDERS)):
            out = await runnable.ainvoke([HumanMessage(f"{TRICKY} {EMAIL} at {ADDRESS}")])
    assert json.loads(out["raw"].content) == EXPECTED


async def test_a_schema_validator_that_rejects_a_placeholder_fails_inside_the_sdk() -> None:
    # A documented limit (redaction.py): the provider's SDK validates the placeholder form, so a
    # constraint that a placeholder breaks (a pattern, EmailStr) raises before this wrapper sees
    # the answer. It is not silently skipped.
    class Checked(BaseModel):
        email: EmailStr

    _, model = wire.wrapped("openai")
    runnable = model.with_structured_output(Checked, method="json_schema", strict=True)
    async with redaction_scope():
        with (
            wire.captured(structured_reply("openai", "json_schema", {"email": "[EMAIL_1]"})),
            pytest.raises(Exception, match="email"),
        ):
            await runnable.ainvoke([HumanMessage(f"Reach me at {EMAIL}")])


def test_structured_output_the_wrapper_cannot_re_point_is_refused_at_construction() -> None:
    inner, model = wire.wrapped("openai")
    with (
        mock.patch.object(
            type(inner), "with_structured_output", return_value=RunnableLambda(lambda x: x)
        ),
        pytest.raises(RedactionUnavailable, match="with_structured_output"),
    ):
        model.with_structured_output(Contact)


# --- streamed structured output ------------------------------------------------------------------

JSON_TEXT = json.dumps(PLACEHOLDERS)
PIECES = [JSON_TEXT[:18], JSON_TEXT[18:40], JSON_TEXT[40:70], JSON_TEXT[70:]]
STREAMED_METHODS = [
    ("openai", wire.openai_stream_text),
    ("openai-responses", wire.responses_stream_text),
]


# langchain-openai puts a parsed pydantic object into additional_kwargs, which pydantic's
# serializer warns about whenever a chunk is dumped (also without the wrapper); not ours.
PARSED_WARNING = pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")


@PARSED_WARNING
@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("schema_kind", ["pydantic", "dict"])
@pytest.mark.parametrize(("provider", "stream"), STREAMED_METHODS)
async def test_streamed_structured_output_returns_the_originals(
    provider: str, stream: Any, schema_kind: str, include_raw: bool, request: pytest.FixtureRequest
) -> None:
    # langchain-openai adds a final chunk that carries the parsed object (and a refusal) in
    # additional_kwargs; it must be restored like the text.
    _, model = wire.wrapped(provider)
    schema: Any = Contact if schema_kind == "pydantic" else DICT_SCHEMA
    runnable = model.with_structured_output(schema, include_raw=include_raw)
    if provider == "openai-responses" and schema_kind == "pydantic":
        # langchain-openai streams a Responses pydantic schema without parsing it, so even the raw
        # provider model raises this ValueError. strict: the case fails once that is supported.
        request.applymarker(pytest.mark.xfail(strict=True, raises=ValueError))
    async with redaction_scope(known={"NAME": [TRICKY], "ADDRESS": [ADDRESS]}):
        with wire.captured(lambda _b: stream(PIECES)) as sent:
            outputs = [
                o async for o in runnable.astream([HumanMessage(f"{TRICKY} {EMAIL} at {ADDRESS}")])
            ]
    for seed in (EMAIL, TRICKY, ADDRESS):
        assert seed not in sent.text
    if include_raw:
        # include_raw streams partial dicts ({"raw": ...}, {"parsed": ...}) in an order that is
        # not fixed, so the parsed one is not always the last.
        parsed = [o["parsed"] for o in outputs if isinstance(o, dict) and "parsed" in o][-1]
    else:
        parsed = outputs[-1]
    assert as_dict(parsed) == EXPECTED


@PARSED_WARNING
async def test_a_graph_node_that_calls_structured_output_under_the_message_stream() -> None:
    from langgraph.graph import END, START, MessagesState, StateGraph

    _, model = wire.wrapped("openai")
    structured = model.with_structured_output(Contact)
    results: list[dict[str, Any]] = []

    async def node(state: MessagesState) -> dict[str, Any]:
        contact = await structured.ainvoke(state["messages"])
        results.append(as_dict(contact))
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("extract", node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", END)
    graph = builder.compile()
    async with redaction_scope(known={"NAME": [TRICKY], "ADDRESS": [ADDRESS]}):
        with wire.captured(lambda _b: wire.openai_stream_text(PIECES)):
            async for _ in graph.astream(
                {"messages": [HumanMessage(f"{TRICKY} {EMAIL} at {ADDRESS}")]},
                stream_mode="messages",
            ):
                pass
    assert results == [EXPECTED]


async def test_a_refusal_block_that_quotes_a_placeholder_is_restored_in_a_stream() -> None:
    _, model = wire.wrapped("openai-responses")
    refusal = "I cannot help [NAME_1] with that."
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.responses_stream_refusal(refusal)):
            chunks = [c async for c in model.astream([HumanMessage(f"I am {TRICKY}")])]
    blocks = [b for c in chunks if isinstance(c.content, list) for b in c.content]
    [block] = [b for b in blocks if isinstance(b, dict) and b.get("type") == "refusal"]
    assert block["refusal"] == f"I cannot help {TRICKY} with that."
