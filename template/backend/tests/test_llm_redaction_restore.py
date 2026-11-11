"""Restoring placeholders (#78): where a raw (JSON-escaped) restore applies, JSON text nested in a
string, and data of any shape. Decided by position in the message, never by a field's name."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.core.redaction import Redactor, redaction_scope
from app.llm.langchain import redaction
from app.llm.langchain.redaction import RedactingChatModel
from tests import wire
from tests.wire import TRICKY

pytestmark = pytest.mark.anyio

REDACTOR = Redactor(wire.DETECTORS)
ADDRESS = '12 "Quoted" Row\\Flat\nLeeds'
KNOWN = {"NAME": [TRICKY], "ADDRESS": [ADDRESS]}
# The history makes both values known to this call, so the placeholders in the answer map back.
HISTORY: list[Any] = [HumanMessage(f"I am {TRICKY} and live at {ADDRESS}")]


async def answered(message: AIMessage) -> AIMessage:
    model = RedactingChatModel(GenericFakeChatModel(messages=iter([message])), REDACTOR)
    async with redaction_scope(known=KNOWN):
        return await model.ainvoke(HISTORY)


# --- raw restore is for the provider's transport strings only ---------------------------------


async def test_a_tool_field_named_arguments_or_partial_json_comes_back_exactly() -> None:
    args = {"arguments": "[NAME_1]", "partial_json": "[NAME_1] at [ADDRESS_1]", "n": 1}
    out = await answered(AIMessage("", tool_calls=[{"name": "run", "args": args, "id": "1"}]))
    assert out.tool_calls[0]["args"] == {
        "arguments": TRICKY,
        "partial_json": f"{TRICKY} at {ADDRESS}",
        "n": 1,
    }


async def test_an_anthropic_tool_use_input_with_those_field_names_comes_back_exactly() -> None:
    block = {
        "type": "tool_use",
        "id": "t1",
        "name": "run",
        "input": {"arguments": "[NAME_1]", "nested": {"partial_json": "[NAME_1]"}},
    }
    out = await answered(AIMessage(content=[block]))
    assert isinstance(out.content, list)
    first = out.content[0]
    assert isinstance(first, dict)
    assert first["input"] == {"arguments": TRICKY, "nested": {"partial_json": TRICKY}}


async def test_transport_strings_are_still_restored_escaped_so_they_stay_valid_json() -> None:
    raw = json.dumps({"who": "[NAME_1]"})
    message = AIMessage(
        "",
        tool_calls=[{"name": "run", "args": {"who": "[NAME_1]"}, "id": "1"}],
        additional_kwargs={
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "run", "arguments": raw}}
            ],
            "function_call": {"name": "run", "arguments": raw},
        },
        invalid_tool_calls=[
            {"name": "run", "args": '{"who": "[NAME_1]", "x', "id": "2", "error": "e"}
        ],
    )
    out = await answered(message)
    expected = {"who": TRICKY}
    assert json.loads(out.additional_kwargs["tool_calls"][0]["function"]["arguments"]) == expected
    assert json.loads(out.additional_kwargs["function_call"]["arguments"]) == expected
    assert json.dumps(TRICKY)[1:-1] in str(out.invalid_tool_calls[0]["args"])


async def test_a_responses_function_call_block_argument_string_is_restored_escaped() -> None:
    block = {
        "type": "function_call",
        "name": "run",
        "arguments": json.dumps({"who": "[NAME_1]"}),
        "call_id": "c1",
    }
    out = await answered(AIMessage(content=[block]))
    assert isinstance(out.content, list)
    first = out.content[0]
    assert isinstance(first, dict)
    assert json.loads(first["arguments"]) == {"who": TRICKY}


# --- JSON text inside a string value is restored at its own level ------------------------------


async def test_json_text_nested_in_a_tool_argument_stays_valid_and_exact() -> None:
    single = json.dumps({"address": "[ADDRESS_1]", "who": "[NAME_1]"})
    double = json.dumps({"inner": json.dumps({"a": "[ADDRESS_1]"}), "n": 1})
    args = {"payload": single, "deep": double, "plain": "[NAME_1]"}
    out = await answered(AIMessage("", tool_calls=[{"name": "run", "args": args, "id": "1"}]))
    got = out.tool_calls[0]["args"]
    assert json.loads(got["payload"]) == {"address": ADDRESS, "who": TRICKY}
    assert json.loads(json.loads(got["deep"])["inner"]) == {"a": ADDRESS}
    assert json.loads(got["deep"])["n"] == 1
    assert got["plain"] == TRICKY


async def test_json_keys_inside_nested_json_text_are_restored_too() -> None:
    payload = json.dumps({"[NAME_1]": "tenant"})
    out = await answered(
        AIMessage("", tool_calls=[{"name": "r", "args": {"p": payload}, "id": "1"}])
    )
    assert json.loads(out.tool_calls[0]["args"]["p"]) == {TRICKY: "tenant"}


async def test_json_text_in_a_structured_output_field_stays_valid_and_exact() -> None:
    _, model = wire.wrapped("openai")
    nested = json.dumps({"a": "[NAME_1]", "inner": json.dumps({"b": "[ADDRESS_1]"})})
    payload = {"name": nested, "email": "x", "address": {"city": "Leeds", "line": "[ADDRESS_1]"}}
    runnable = model.with_structured_output(wire.Contact, method="function_calling")
    async with redaction_scope(known=KNOWN):
        with wire.captured(lambda _b: wire.tool_reply("openai", "Contact", payload)):
            out = await runnable.ainvoke(HISTORY)
    value = json.loads(out.name)
    assert value["a"] == TRICKY
    assert json.loads(value["inner"]) == {"b": ADDRESS}
    assert out.address.line == ADDRESS


# --- data of any shape ----------------------------------------------------------------------


ODD_TYPES = [["flat", "house"], {"a": 1}, 5, None, 2.5, True]


@pytest.mark.parametrize("odd", ODD_TYPES, ids=repr)
async def test_a_type_field_of_any_shape_does_not_break_restoring(odd: Any) -> None:
    args = {"type": odd, "who": "[NAME_1]", "nested": {"type": odd, "k": [{"type": odd}]}}
    out = await answered(AIMessage("", tool_calls=[{"name": "r", "args": args, "id": "1"}]))
    got = out.tool_calls[0]["args"]
    assert got["type"] == odd
    assert got["who"] == TRICKY
    assert got["nested"] == {"type": odd, "k": [{"type": odd}]}


@pytest.mark.parametrize("odd", ODD_TYPES, ids=repr)
def test_a_type_field_of_any_shape_does_not_break_redacting(odd: Any) -> None:
    args = {"type": odd, "nested": {"type": odd}}
    messages = [
        HumanMessage(content=[{"type": odd, "text": "hello"}]),
        AIMessage("", tool_calls=[{"name": "r", "args": args, "id": "1"}]),
    ]
    prepared = redaction.prepare(REDACTOR, messages)
    assert prepared.messages[1].tool_calls[0]["args"] == args  # type: ignore[attr-defined]


async def test_the_untouched_block_check_is_for_content_blocks_only() -> None:
    # A tool argument that happens to look like a thinking block is plain data and is restored.
    args = {"type": "thinking", "thinking": "[NAME_1]"}
    out = await answered(AIMessage("", tool_calls=[{"name": "r", "args": args, "id": "1"}]))
    assert out.tool_calls[0]["args"] == {"type": "thinking", "thinking": TRICKY}
