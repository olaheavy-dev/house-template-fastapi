"""Tool calls through the redacting chat model (#78): the model is shown placeholders in its tool
arguments, and the app gets the originals back, whichever shape the provider uses (a parsed
`args` dict, Anthropic's `tool_use` block, streamed argument fragments, raw argument strings)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, cast
from unittest import mock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding, RunnableLambda
from langchain_core.tools import tool

from app.llm.core.redaction import Redactor, redaction_scope
from app.llm.langchain import redaction
from app.llm.langchain.redaction import RedactingChatModel, RedactionUnavailable
from tests import wire
from tests.test_llm_redaction_context import StreamingHandler
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


# --- streamed arguments that hold JSON text inside a string (#102) ------------------------------

# A quote, a backslash, a newline and non-ASCII text: every kind of character a JSON string escapes.
HARD = 'Jo "JB" O\\Brien\nna\u00efve \u00e9'
STREAMED = ["openai", "openai-responses", "anthropic"]


def nested(levels: int, inner: dict[str, Any]) -> str:
    """The model's argument string: `inner` as JSON text, put inside a string value `levels` times
    (0: the plain argument)."""
    text = json.dumps(inner)
    for _ in range(levels):
        text = json.dumps({"payload": text})
    return text


def unwrap(arguments: str, levels: int) -> Any:
    """What the tool does with its argument: read the JSON, then the JSON text inside it."""
    data = json.loads(arguments)
    for _ in range(levels):
        data = json.loads(data["payload"])
    return data


def stream_data(provider: str, pieces: list[str]) -> bytes:
    if provider == "openai":
        return wire.openai_stream_tool("lookup", pieces)
    if provider == "openai-responses":
        return wire.responses_stream_tool("lookup", pieces)
    deltas = [{"type": "input_json_delta", "partial_json": piece} for piece in pieces]
    start = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    return wire.anthropic_stream([(start, deltas)])


def split_inside_placeholder(text: str) -> list[str]:
    cut = text.index("[NAME_1]") + 4
    return [text[:cut], text[cut:]]


def argument_strings(chunks: list[Any]) -> dict[str, str]:
    """The argument fragments of a stream, joined by where they travelled."""
    found: dict[str, list[str]] = {"tool_call_chunks": [], "partial_json": [], "arguments": []}
    for chunk in chunks:
        found["tool_call_chunks"] += [c["args"] or "" for c in chunk.tool_call_chunks]
        if isinstance(chunk.content, list):
            for block in chunk.content:
                for key in ("partial_json", "arguments"):
                    if isinstance(block, dict) and isinstance(block.get(key), str):
                        found[key].append(block[key])
    return {key: "".join(parts) for key, parts in found.items() if parts}


@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("provider", STREAMED)
async def test_streamed_arguments_with_json_text_inside_are_restored_at_their_level(
    provider: str, levels: int
) -> None:
    text = nested(levels, {"who": "[NAME_1]", "n": 3})
    expected = {"who": HARD, "n": 3}
    _, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [HARD]}):
        with wire.captured(lambda _b: stream_data(provider, split_inside_placeholder(text))):
            chunks = [c async for c in bound.astream([HumanMessage(f"Find {HARD}")])]
    strings = argument_strings(chunks)
    assert strings  # the arguments travelled somewhere
    for joined in strings.values():
        assert "[NAME_1]" not in joined
        assert unwrap(joined, levels) == expected
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    [call] = total.tool_calls
    assert unwrap(json.dumps(call["args"]), levels) == expected
    # The merged message carries the restored string once, in the block the provider uses for it,
    # with no stray block beside it (a tail without its index would add one).
    if provider == "anthropic":
        [block] = total.content
        assert block["type"] == "tool_use"
        assert block["partial_json"] == strings["partial_json"]
        assert unwrap(block["partial_json"], levels) == expected
    elif provider == "openai-responses":
        [block] = total.content
        assert block["type"] == "function_call"
        assert block["arguments"] == strings["arguments"]
        assert unwrap(block["arguments"], levels) == expected
    else:
        assert total.content in ("", [])


@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("provider", STREAMED)
async def test_ainvoke_with_a_streaming_handler_restores_nested_arguments_too(
    provider: str, levels: int
) -> None:
    """LangGraph's message stream makes ainvoke take the streaming branch of the model."""
    text = nested(levels, {"who": "[NAME_1]"})
    _, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [HARD]}):
        with wire.captured(lambda _b: stream_data(provider, split_inside_placeholder(text))):
            answer = await bound.ainvoke(
                [HumanMessage(f"Find {HARD}")], config={"callbacks": [StreamingHandler()]}
            )
    [call] = answer.tool_calls
    assert unwrap(json.dumps(call["args"]), levels) == {"who": HARD}


async def test_a_plain_streamed_argument_is_restored_exactly_and_keeps_its_name_and_id() -> None:
    pieces = ['{"who": "[NA', 'ME_1]", "mail": "[EMAIL', '_1]"}']
    _, model = wire.wrapped("openai")
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_stream_tool("lookup", pieces)):
            chunks = [c async for c in bound.astream([HumanMessage(f"Find {TRICKY} at {EMAIL}")])]
    joined = argument_strings(chunks)["tool_call_chunks"]
    assert joined == json.dumps({"who": TRICKY, "mail": EMAIL})
    # The fragments are held, so the first chunks carry empty arguments and the call keeps its name
    # and id from the first chunk.
    first = chunks[0].tool_call_chunks[0]
    assert (first["name"], first["id"], first["args"]) == ("lookup", "call_1", "")


async def test_text_before_a_streamed_tool_call_arrives_before_the_held_arguments() -> None:
    pieces = ['{"who": "[NAME_1]"}']
    text_start = {"type": "text", "text": ""}
    text_deltas = [{"type": "text_delta", "text": "Looking up [NAME_1]. "}]
    tool_start = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    tool_deltas = [{"type": "input_json_delta", "partial_json": p} for p in pieces]
    data = wire.anthropic_stream([(text_start, text_deltas), (tool_start, tool_deltas)])
    _, model = wire.wrapped("anthropic")
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: data):
            chunks = [c async for c in bound.astream([HumanMessage(f"Find {TRICKY}")])]
    with_text = [i for i, c in enumerate(chunks) if TRICKY in c.text]
    with_args = [i for i, c in enumerate(chunks) if any(x["args"] for x in c.tool_call_chunks)]
    assert with_text
    assert with_args
    assert with_text[0] < with_args[0]


def parallel_stream() -> bytes:
    """An OpenAI stream of two tool calls (index 0 and 1) whose fragments are interleaved."""
    first = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            },
            {
                "index": 1,
                "id": "call_b",
                "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            },
        ],
    }

    def piece(index: int, text: str) -> tuple[str | None, Any]:
        call = {"index": index, "function": {"arguments": text}}
        return (None, wire._chunk({"tool_calls": [call]}))

    events: list[tuple[str | None, Any]] = [(None, wire._chunk(first))]
    events += [
        piece(0, '{"who": "[NA'),
        piece(1, '{"who": "pl'),
        piece(0, 'ME_1]"}'),
        piece(1, 'ain"}'),
        (None, wire._chunk({}, "tool_calls")),
        (None, wire._usage_chunk()),
        (None, "[DONE]"),
    ]
    return wire.sse(events)


async def test_two_interleaved_parallel_tool_calls_each_keep_their_own_arguments() -> None:
    _, model = wire.wrapped("openai")
    bound = model.bind_tools([lookup])
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: parallel_stream()):
            chunks = [c async for c in bound.astream([HumanMessage(f"Find {TRICKY}")])]
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    calls = {c["id"]: c["args"] for c in total.tool_calls}
    assert calls == {"call_a": {"who": TRICKY}, "call_b": {"who": "plain"}}
    assert not total.invalid_tool_calls


def test_whole_tool_calls_without_an_index_each_keep_their_own_restored_arguments() -> None:
    """Ollama, Gemini, Bedrock and OpenAI-compatible endpoints send whole calls with no index:
    LangChain never merges them, so they must not share one buffer."""
    from app.llm.langchain.redaction import _StreamRestorers

    restorers = _StreamRestorers({"[NAME_1]": TRICKY}, json_text=False)

    def call(call_id: str, args: str) -> dict[str, Any]:
        return {
            "name": "lookup",
            "id": call_id,
            "args": args,
            "index": None,
            "type": "tool_call_chunk",
        }

    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[call("c1", '{"who": "[NAME_1]"}'), call("c2", '{"who": "y"}')],
    )
    out = restorers.chunk(chunk)
    assert [(c["id"], json.loads(c["args"] or "")) for c in out.tool_call_chunks] == [
        ("c1", {"who": TRICKY}),
        ("c2", {"who": "y"}),
    ]
    assert restorers.flush() == []  # no phantom call at the end


async def test_a_consumer_that_stops_mid_tool_call_is_given_no_arguments() -> None:
    pieces = ['{"who": "[NAME_1]", "mail": "[EMAIL_', '1]"}']
    _, model = wire.wrapped("openai")
    bound = model.bind_tools([lookup])
    seen: list[AIMessageChunk] = []
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_stream_tool("lookup", pieces)):
            stream = cast(
                AsyncGenerator[AIMessageChunk, None],
                bound.astream([HumanMessage(f"Find {TRICKY}")]),
            )
            async with aclosing(stream) as chunks:
                async for chunk in chunks:
                    seen.append(chunk)
                    break
    assert seen
    assert not any(c["args"] for chunk in seen for c in chunk.tool_call_chunks)
    assert "[NAME_1]" not in repr(seen)


async def test_an_argument_with_a_placeholder_the_model_invented_stays_and_is_counted() -> None:
    from structlog.testing import capture_logs

    _, model = wire.wrapped("openai")
    bound = model.bind_tools([lookup])
    pieces = ['{"who": "[NAME_1]", "mail": "[EMAIL_', '9]"}']
    with capture_logs() as logs:
        async with redaction_scope(known={"NAME": [TRICKY]}):
            with wire.captured(lambda _b: wire.openai_stream_tool("lookup", pieces)):
                chunks = [c async for c in bound.astream([HumanMessage(f"Find {TRICKY}")])]
    joined = argument_strings(chunks)["tool_call_chunks"]
    assert json.loads(joined) == {"who": TRICKY, "mail": "[EMAIL_9]"}
    [event] = [e for e in logs if e["event"] == "llm_unrestored_placeholders"]
    assert event["count"] == 1


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
    cut_args = '{"mail": "' + EMAIL + '", "who":'
    message = AIMessage(
        "",
        invalid_tool_calls=[
            {
                "name": "lookup",
                "args": cut_args,
                "id": "1",
                "error": "cut",
            }
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
                {"name": "lookup", "args": '{"mail": "' + EMAIL + '"', "id": "c1", "error": "cut"}
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


def test_a_media_block_is_sent_exactly_as_it_was_even_when_its_text_looks_personal() -> None:
    # Media is counted, not scanned: a url that holds an email is the provider's to fetch, not ours
    # to rewrite (rewriting would break it).
    image = {"type": "image", "source": {"type": "url", "url": f"https://example.com/p?m={EMAIL}"}}
    message = HumanMessage(content=[{"type": "text", "text": "Look"}, image])
    prepared = redaction.prepare(Redactor(wire.DETECTORS), [message])
    [sent] = prepared.messages
    assert isinstance(sent.content, list)
    assert sent.content[1] == image
    assert prepared.count == 0
    assert prepared.unscanned == 1


def lookup_lead(who: str) -> str:
    """Look up a lead. Leads for jo.bloggs@example.com go to the office."""
    return who


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_tools_definition_reaches_the_wire_with_placeholders_and_its_name_as_it_is(
    provider: str,
) -> None:
    # #112's review: a tool's description is text the provider reads. The email in the definition
    # and in the message is one value, so both are [EMAIL_1]; the answer gets it back.
    _, model = wire.wrapped(provider)
    with wire.captured(lambda _b: wire.text_reply(provider, "Ask [EMAIL_1]")) as sent:
        answer = await model.bind_tools([lookup_lead]).ainvoke(
            [HumanMessage(f"Is {EMAIL} a lead?")]
        )
    assert EMAIL not in sent.text
    assert sent.text.count("[EMAIL_1]") == 2
    assert '"lookup_lead"' in sent.text
    assert answer.text == f"Ask {EMAIL}"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_tool_binding_with_nothing_to_redact_sends_the_same_definition_objects(
    provider: str,
) -> None:
    # Nothing to replace: the bound definitions are passed on as the very objects bound, so the
    # request is byte for byte what the provider model alone would send.
    inner, model = wire.wrapped(provider)
    bound = model.bind_tools([lookup])
    assert isinstance(bound, RunnableBinding)
    sent: list[Any] = []
    real = type(inner)._agenerate

    async def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        sent.append(kwargs.get("tools"))
        return await real(self, *args, **kwargs)

    reply = wire.tool_reply(provider, "lookup", {"who": "x", "mail": "y"})
    with mock.patch.object(type(inner), "_agenerate", spy), wire.captured(lambda _b: reply):
        await bound.ainvoke([HumanMessage("find the property")])
    assert sent == [bound.kwargs["tools"]]
    assert sent[0] is bound.kwargs["tools"]


@pytest.mark.parametrize("provider", wire.STREAMING_PROVIDERS)
async def test_a_streamed_call_sends_its_tools_definition_with_placeholders(provider: str) -> None:
    from tests.test_llm_redaction_stream import stream_reply

    _, model = wire.wrapped(provider)
    with wire.captured(stream_reply(provider, ["Ask ", "[EMAIL_1]"])) as sent:
        chunks = [
            chunk
            async for chunk in model.bind_tools([lookup_lead]).astream(
                [HumanMessage(f"Is {EMAIL} a lead?")]
            )
        ]
    assert EMAIL not in sent.text
    assert sent.text.count("[EMAIL_1]") == 2
    assert "".join(chunk.text for chunk in chunks) == f"Ask {EMAIL}"
