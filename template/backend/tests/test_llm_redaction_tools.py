"""Tool calls through the redacting chat model (#78): the model is shown placeholders in its tool
arguments, and the app gets the originals back, whichever shape the provider uses (a parsed
`args` dict, Anthropic's `tool_use` block, streamed argument fragments, raw argument strings)."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding, RunnableLambda
from langchain_core.tools import tool

from app.llm.core.redaction import Redactor, redaction_scope
from app.llm.langchain import redaction
from app.llm.langchain.redaction import RedactingChatModel, RedactionUnavailable
from tests import wire
from tests.wire import EMAIL, PROVIDERS, TRICKY

pytestmark = pytest.mark.anyio


@tool
def lookup(who: str, mail: str) -> str:
    """Look a person up."""
    return "found"


def tool_call_id(provider: str) -> str:
    return "toolu_1" if provider == "anthropic" else "call_1"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_tool_call_arguments_come_back_as_the_originals_in_every_shape(
    provider: str,
) -> None:
    _, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup], tool_choice="lookup")
    reply = wire.tool_reply(provider, "lookup", {"who": "[NAME_1]", "mail": "[EMAIL_1]"})
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: reply) as sent:
            answer = await bound.ainvoke([HumanMessage(f"Find {TRICKY} at {EMAIL}")])
    assert TRICKY not in sent.text
    assert EMAIL not in sent.text
    assert answer.tool_calls[0]["args"] == {"who": TRICKY, "mail": EMAIL}
    if provider == "anthropic":
        block = next(b for b in answer.content if isinstance(b, dict) and b["type"] == "tool_use")
        assert block["input"] == {"who": TRICKY, "mail": EMAIL}
    else:
        # langchain-openai keeps no raw argument string on the message, so nothing is left over.
        assert "tool_calls" not in answer.additional_kwargs
    assert "[NAME_1]" not in repr(answer)
    assert "[EMAIL_1]" not in repr(answer)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_the_next_call_redacts_the_history_that_now_holds_the_originals(
    provider: str,
) -> None:
    # Test 19 and 20: the answer's arguments hold a name with a quote and a backslash. They are
    # sent back as a tool call in the history, with the tool's result, and nothing leaks.
    _, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup])
    reply = wire.tool_reply(provider, "lookup", {"who": "[NAME_1]", "mail": "[EMAIL_1]"})
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: reply):
            answer = await bound.ainvoke([HumanMessage(f"Find {TRICKY} at {EMAIL}")])
        history = [
            HumanMessage(f"Find {TRICKY} at {EMAIL}"),
            answer,
            ToolMessage(f"{TRICKY} {EMAIL}", tool_call_id=tool_call_id(provider)),
        ]
        with wire.captured(lambda _b: wire.text_reply(provider, "done")) as sent:
            await bound.ainvoke(history)
    assert TRICKY not in sent.text
    assert json.dumps(TRICKY)[1:-1] not in sent.text  # nor in its JSON-escaped form
    assert EMAIL not in sent.text
    # Every body is valid JSON: the placeholders are plain text inside it.
    assert sent.json_bodies()


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_binding_options_and_a_request_with_nothing_to_redact_match_the_raw_binding(
    provider: str,
) -> None:
    inner, model = wire.wrapped(provider)
    messages = [HumanMessage("find the property")]
    options: dict[str, Any] = {"tool_choice": "lookup"}
    if provider != "anthropic":
        options["strict"] = True
    reply = wire.tool_reply(provider, "lookup", {"who": "x", "mail": "y"})
    with wire.captured(lambda _b: reply) as raw:
        await inner.bind_tools([lookup], **options).ainvoke(messages)
    with wire.captured(lambda _b: reply) as wrapped:
        await model.bind_tools([lookup], **options).ainvoke(messages)
    assert wrapped.bodies == raw.bodies
    body = wrapped.json_bodies()[0]
    assert body["tools"]
    if provider == "openai":
        assert body["tools"][0]["function"]["strict"] is True
        assert body["tool_choice"]["function"]["name"] == "lookup"
    if provider == "openai-responses":
        assert body["tools"][0]["strict"] is True


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_streamed_tool_arguments_split_across_chunks_are_restored(provider: str) -> None:
    pieces = ['{"who": "[NA', 'ME_1]", "mail": "[EMAIL', '_1]"}']
    if provider == "openai":
        data = wire.openai_stream_tool("lookup", pieces)
    else:
        deltas = [{"type": "input_json_delta", "partial_json": piece} for piece in pieces]
        start = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
        data = wire.anthropic_stream([(start, deltas)])
    _, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: data) as sent:
            chunks = [c async for c in bound.astream([HumanMessage(f"Find {TRICKY} at {EMAIL}")])]
    assert TRICKY not in sent.text
    assert EMAIL not in sent.text
    # No fragment shows a placeholder, and the fragments together are the originals as valid JSON.
    fragments = "".join(c["args"] or "" for chunk in chunks for c in chunk.tool_call_chunks)
    assert "[" not in fragments
    assert json.loads(fragments) == {"who": TRICKY, "mail": EMAIL}
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    assert total.tool_calls[0]["args"] == {"who": TRICKY, "mail": EMAIL}


# --- raw argument strings, through a scripted model --------------------------------------------


def scripted(message: AIMessage) -> RedactingChatModel:
    inner = GenericFakeChatModel(messages=iter([message]))
    return RedactingChatModel(inner, Redactor(wire.DETECTORS))


def raw_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def test_raw_argument_strings_are_restored_as_valid_json() -> None:
    args = {"who": "[NAME_1]", "mail": "[EMAIL_1]"}
    raw_args = json.dumps(args)
    scripted_answer = AIMessage(
        "",
        tool_calls=[{"name": "lookup", "args": args, "id": "1"}],
        additional_kwargs={
            "tool_calls": [raw_call("1", "lookup", raw_args)],
            "function_call": {"name": "lookup", "arguments": raw_args},
        },
        invalid_tool_calls=[
            {"name": "lookup", "args": '{"who": "[NAME_1]", "mail": ', "id": "2", "error": "cut"}
        ],
    )
    model = scripted(scripted_answer)
    async with redaction_scope(known={"NAME": [TRICKY]}):
        answer = await model.ainvoke([HumanMessage(f"Find {TRICKY} at {EMAIL}")])
    expected = {"who": TRICKY, "mail": EMAIL}
    assert answer.tool_calls[0]["args"] == expected
    # The older raw argument strings: valid JSON again, with the quote and backslash escaped.
    [raw_call_back] = answer.additional_kwargs["tool_calls"]
    assert json.loads(raw_call_back["function"]["arguments"]) == expected
    assert json.loads(answer.additional_kwargs["function_call"]["arguments"]) == expected
    # Text cut off mid-document is not JSON: its placeholder is still put back, escaped.
    [invalid] = answer.invalid_tool_calls
    assert invalid["args"] is not None
    assert json.dumps(TRICKY)[1:-1] in invalid["args"]


def test_raw_argument_strings_in_the_history_are_redacted_or_left_alone() -> None:
    raw_with = json.dumps({"who": "Jo", "mail": EMAIL})
    raw_without = '{"city": "Leeds", "n": 3}'
    message = AIMessage(
        "",
        additional_kwargs={
            "tool_calls": [raw_call("1", "a", raw_with), raw_call("2", "b", raw_without)]
        },
    )
    prepared = redaction.prepare(Redactor(wire.DETECTORS), [message])
    [sent] = prepared.messages
    first, second = sent.additional_kwargs["tool_calls"]
    assert EMAIL not in first["function"]["arguments"]
    assert json.loads(first["function"]["arguments"]) == {"who": "Jo", "mail": "[EMAIL_1]"}
    assert second["function"]["arguments"] == raw_without  # untouched, byte for byte
    # The email is in the raw string, and AIMessage also parses it into tool_calls: two places.
    assert prepared.count == 2


# --- fail closed at construction -----------------------------------------------------------------


def test_a_tool_binding_that_is_not_a_plain_binding_of_the_model_is_refused() -> None:
    class Odd(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return RunnableLambda(lambda x: x)  # not a binding of this model: not redactable

    model = RedactingChatModel(Odd(messages=iter([])), Redactor(wire.DETECTORS))
    with pytest.raises(RedactionUnavailable, match="plain binding"):
        model.bind_tools([lookup])


def test_a_model_that_takes_no_tools_binds_to_the_wrapper_itself() -> None:
    # A scripted test double returns itself from bind_tools; the wrapper then has nothing to bind.
    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    model = RedactingChatModel(Fake(messages=iter([])), Redactor(wire.DETECTORS))
    assert model.bind_tools([lookup]) is model


# --- shapes that are easy to miss -----------------------------------------------------------------


def test_invalid_tool_calls_in_the_history_are_redacted() -> None:
    message = AIMessage(
        "",
        invalid_tool_calls=[
            {"name": "lookup", "args": f'{{"mail": "{EMAIL}", "who":', "id": "1", "error": "cut"}
        ],
    )
    prepared = redaction.prepare(Redactor(wire.DETECTORS), [message])
    [sent] = prepared.messages
    assert isinstance(sent, AIMessage)
    [invalid] = sent.invalid_tool_calls
    assert invalid["args"] is not None
    assert EMAIL not in invalid["args"]
    assert "[EMAIL_1]" in invalid["args"]
    assert prepared.count == 1


async def test_invalid_tool_calls_in_the_history_do_not_reach_the_wire() -> None:
    _, model = wire.wrapped("openai")
    history = [
        HumanMessage("hi"),
        AIMessage(
            "",
            invalid_tool_calls=[
                {"name": "lookup", "args": f'{{"mail": "{EMAIL}"', "id": "c1", "error": "cut"}
            ],
        ),
        ToolMessage("bad arguments", tool_call_id="c1"),
    ]
    with wire.captured(lambda _b: wire.openai_text("ok")) as sent:
        await model.ainvoke(history)
    assert EMAIL not in sent.text


def test_the_older_function_call_argument_string_in_the_history_is_redacted() -> None:
    arguments = json.dumps({"mail": EMAIL})
    message = AIMessage(
        "", additional_kwargs={"function_call": {"name": "lookup", "arguments": arguments}}
    )
    prepared = redaction.prepare(Redactor(wire.DETECTORS), [message])
    [sent] = prepared.messages
    sent_arguments = sent.additional_kwargs["function_call"]["arguments"]
    assert EMAIL not in sent_arguments
    assert json.loads(sent_arguments) == {"mail": "[EMAIL_1]"}


def test_a_tool_result_block_has_its_text_redacted_and_its_image_counted_and_left_alone() -> None:
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    block = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "text", "text": f"Mail {EMAIL}"}, image],
    }
    message = HumanMessage(content=[block])
    prepared = redaction.prepare(Redactor(wire.DETECTORS), [message])
    [sent] = prepared.messages
    assert isinstance(sent.content, list)
    [sent_block] = sent.content
    assert isinstance(sent_block, dict)
    assert sent_block["tool_use_id"] == "toolu_1"
    assert sent_block["content"][0]["text"] == "Mail [EMAIL_1]"
    assert sent_block["content"][1] == image  # not walked: its data is sent as it was
    assert prepared.unscanned == 1
    assert prepared.count == 1


def test_a_structured_runnable_that_is_too_deep_to_check_is_refused_fail_closed() -> None:
    # After the re-pointing nothing reaches the provider model, but a binding chain deeper than any
    # LangChain builds cannot be proved clean, so the wrapper refuses it.
    inner, model = wire.wrapped("openai")
    deep: Any = inner
    for _ in range(15):
        deep = RunnableBinding(bound=deep, kwargs={})
    with (
        mock.patch.object(type(inner), "with_structured_output", return_value=deep),
        pytest.raises(RedactionUnavailable, match="with_structured_output"),
    ):
        model.with_structured_output(wire.Contact)
