"""Personal-data redaction around a LangChain chat model (#78, docs/decisions/0006).

`RedactingChatModel` wraps the gateway's provider model. Before a request goes out it replaces
contact, bank and ID numbers and the values the app named (`redaction_scope`) by placeholders such
as [EMAIL_1]; after the answer comes back it puts the originals back. Code keeps calling
`gateway.model()` as before: ainvoke, astream, bind_tools, with_structured_output and agents all
go through it. What it does not do: find every kind of personal data, or turn the data into
anonymous data (the mapping is the key, and it lives in memory for one call).

Why a wrapper and not a subclass. BaseChatModel fires its callbacks around `_agenerate`, so a
subclass of ChatOpenAI would show the budget guard and the tracer the originals, or show
LangGraph's message stream the placeholders. The wrapper is its own chat model, so what callers
see (its run, its tokens, its answer) is restored text, and the model inside is called as a
separate, hidden run that carries the budget guard and the gateway's tracer with redacted text:

- The inner call gets only its own callbacks plus the parent's LangChainTracers. Other parent
  handlers, such as LangGraph's message stream, never see it, so no placeholder reaches a user and
  no token is sent twice. It is also tagged `nostream`, which LangGraph's message stream skips.
- The number of replaced values travels as call metadata (`redacted_count`, a count, never a
  value) to BudgetGuard, which writes it on the call's llm_call row. Nothing is stored on the
  wrapper, so concurrent requests cannot mix their counts or mappings.

What changes for code that held the model: `isinstance(gateway.model(), ChatOpenAI)` is False now
(`_llm_type`, `profile`, `model_name` and `_get_ls_params` are forwarded, and the request body is
unchanged when nothing matches). Async only: the sync methods raise, because LangChain swallows
errors from async callbacks on the sync path, which would skip the budget as well.

Limits worth knowing:
- Text from the model's thinking and reasoning blocks is neither redacted on the way out nor
  restored on the way back (a provider checks their signature), so it can show placeholders.
- A schema validator that rejects a placeholder (EmailStr, a pattern) fails inside the provider's
  SDK before the answer reaches this wrapper. Restored values are not validated again.
- In a stream, text, tool-call arguments (also the Responses API's repeated `function_call`
  blocks and the older `function_call`/`tool_calls` argument strings), Anthropic's `partial_json`,
  refusals, and the parsed object of a structured answer in `additional_kwargs` are restored. What
  stays as the model wrote it: thinking and reasoning blocks (including `additional_kwargs`
  `reasoning`).
- Images and other media are not scanned (they are counted in `llm_unscanned_images`). A message's
  `name` field cannot take a placeholder (a provider accepts only plain names there), so a call
  whose `name` matches a detector or a known value is refused, nothing sent.
- A LangChainTracer you pass in a call's config with a client of its own records what that client
  is given; only the gateway's client masks.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

import structlog
from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.tracers.langchain import LangChainTracer
from pydantic import BaseModel, PrivateAttr

from app.llm.core import metrics
from app.llm.core.redaction import RedactionFailed, Redactor, StreamRestorer, restore

__all__ = ["RedactingChatModel", "RedactionUnavailable", "masking_anonymizer"]

log = structlog.get_logger()

# Content blocks that are not text: counted, and sent as they are.
_MEDIA = frozenset(
    {"image", "image_url", "input_image", "audio", "input_audio", "file", "input_file", "video"}
)
# Blocks the provider checks or the model produced from placeholders: never changed.
_UNTOUCHED = frozenset({"thinking", "redacted_thinking", "reasoning"})
# String values of a content block that are structure, not text.
_STRUCTURAL = frozenset(
    {
        "type",
        "id",
        "signature",
        "data",
        "encrypted_content",
        "call_id",
        "tool_use_id",
        "name",
        "media_type",
        "mime_type",
        "role",
        "status",
        "format",
        "detail",
    }
)
# Keys whose string value is raw JSON text (a provider's argument string, a streamed fragment).
_RAW_KEYS = frozenset({"arguments", "partial_json"})

# What the walkers call for each string: (the text, the field name in front of it) -> the text. The
# field name ("date of birth: ") is context for detection only; see Redactor.redact(contexts=).
Leaf = Callable[[str, str], str]


class RedactionUnavailable(RuntimeError):
    """The wrapper cannot guarantee that every request it builds is redacted, so it refuses to be
    built (fail closed). Raised when a LangChain method returns a shape this code does not
    know."""


# --- walking the strings of a message ----------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# A dict that pairs a label with a `value` ("label": "Date of birth", "value": "12/03/1980").
_LABEL_KEYS = ("label", "name", "field", "key", "title")


def _field_context(key: Any) -> str:
    """A field name as words in front of its value: `date_of_birth`, `dateOfBirth` and
    `NHSNumber` become "date of birth: " and "NHS Number: ", so a rule that needs a word before the
    value ("date of birth", "account", "sort code", "NHS") finds it when the value sits in a field
    of that name."""
    if not isinstance(key, str) or not key:
        return ""
    words = _CAMEL.sub(" ", key).replace("_", " ").replace("-", " ").replace(".", " ")
    return " ".join(words.split())[-40:] + ": "


def _label_of(pairs: Any) -> str | None:
    """The text that labels a `value` in a dict of (key, value) pairs, if there is one."""
    for key, value in pairs:
        key = key.value if isinstance(key, _Str) else key
        text = value.value if isinstance(value, _Str) else value
        if key in _LABEL_KEYS and isinstance(text, str) and text.strip():
            return text
    return None


def _value_context(key: Any, label: str | None) -> str:
    return _field_context(label) if key == "value" and label else _field_context(key)


def _walk_data(obj: Any, leaf: Leaf, context: str = "") -> Any:
    """Every string value in plain data (tool arguments, a parsed answer), keys untouched. A value
    carries its field name as context (a list item, its list's; a `value` its label's), and a
    string that is JSON text is walked as JSON text."""
    if isinstance(obj, str):
        return _walk_text(obj, leaf, context)
    if isinstance(obj, dict):
        label = _label_of(obj.items())
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            # The key is text too (an email or a phone number can be a key): the key first, then
            # its value, the same order the scanner uses for JSON text.
            new_key = leaf(key, context) if isinstance(key, str) else key
            out[new_key] = _walk_data(value, leaf, _value_context(key, label))
        return out
    if isinstance(obj, list | tuple):
        return [_walk_data(value, leaf, context) for value in obj]
    return obj


class _Str:
    """A JSON string token: where it is in the text, and what it says."""

    __slots__ = ("end", "start", "value")

    def __init__(self, start: int, end: int, value: str) -> None:
        self.start, self.end, self.value = start, end, value


class _Obj:
    __slots__ = ("members",)

    def __init__(self, members: list[tuple[_Str, Any]]) -> None:
        self.members = members  # (key token, value), in order, duplicate keys kept


class _Arr:
    __slots__ = ("items",)

    def __init__(self, items: list[Any]) -> None:
        self.items = items


def _scan(text: str) -> Any:
    """The structure of JSON text that has already been parsed by json.loads (so it is valid),
    with the position of every string token. A number, true, false or null is not kept: it is never
    changed, and the text around a replaced string is never rewritten."""
    position = 0

    def skip() -> None:
        nonlocal position
        while text[position] in " \t\r\n":
            position += 1

    def string() -> _Str:
        nonlocal position
        start = position
        position += 1
        while text[position] != '"':
            position += 2 if text[position] == "\\" else 1
        position += 1
        return _Str(start, position, json.loads(text[start:position]))

    def value() -> Any:
        nonlocal position
        skip()
        char = text[position]
        if char == '"':
            return string()
        if char == "{":
            position += 1
            members: list[tuple[_Str, Any]] = []
            skip()
            if text[position] == "}":
                position += 1
                return _Obj(members)
            while True:
                skip()
                key = string()
                skip()
                position += 1  # the colon
                members.append((key, value()))
                skip()
                char = text[position]
                position += 1
                if char == "}":
                    return _Obj(members)
        if char == "[":
            position += 1
            items: list[Any] = []
            skip()
            if text[position] == "]":
                position += 1
                return _Arr(items)
            while True:
                items.append(value())
                skip()
                char = text[position]
                position += 1
                if char == "]":
                    return _Arr(items)
        while position < len(text) and text[position] not in ",]} \t\r\n":
            position += 1  # a number or a literal
        return None

    return value()


def _visit(node: Any, leaf: Leaf, context: str, edits: list[tuple[int, int, str]]) -> None:
    if isinstance(node, _Str):
        new = _walk_text(node.value, leaf, context)
        if new != node.value:
            edits.append((node.start, node.end, json.dumps(new, ensure_ascii=False)))
    elif isinstance(node, _Obj):
        label = _label_of(node.members)
        for key, child in node.members:
            changed = leaf(key.value, context)  # the key is text too, with its parent's context
            if changed != key.value:
                edits.append((key.start, key.end, json.dumps(changed, ensure_ascii=False)))
            _visit(child, leaf, _value_context(key.value, label), edits)
    elif isinstance(node, _Arr):
        for child in node.items:
            _visit(child, leaf, context, edits)


def _walk_text(text: str, leaf: Leaf, context: str = "") -> str:
    """One string. JSON text (an object or array: an argument string, a tool result) is read, its
    string values are walked with their field names, and only those string tokens that changed are
    written again, so numbers, whitespace, key order and duplicate keys stay exactly as they were.
    Reading matters: a value with a quote or a backslash is escaped in JSON text and would not match
    as it is. JSON nested too deeply to read raises RecursionError, which prepare turns into a
    refusal. Anything else is one string."""
    if text.lstrip()[:1] in ("{", "["):
        try:
            parsed = json.loads(text)
        except ValueError:
            return leaf(text, context)
        if isinstance(parsed, dict | list):
            edits: list[tuple[int, int, str]] = []
            _visit(_scan(text), leaf, context, edits)
            # Edits are in document order (keys and values as the scanner meets them): one pass of
            # slices, so many replacements stay linear.
            pieces: list[str] = []
            position = 0
            for start, end, replacement in edits:
                pieces.append(text[position:start])
                pieces.append(replacement)
                position = end
            pieces.append(text[position:])
            return "".join(pieces)
    return leaf(text, context)


_walk_raw = _walk_text  # the older name: raw JSON argument strings


def _type_of(block: dict[str, Any]) -> str | None:
    """A content block's type if it is a string (user data can put anything in a `type`)."""
    kind = block.get("type")
    return kind if isinstance(kind, str) else None


def _string_tokens(node: Any) -> Iterator[_Str]:
    """Every string token of a scanned JSON text, keys and values, in document order."""
    if isinstance(node, _Str):
        yield node
    elif isinstance(node, _Obj):
        for key, child in node.members:
            yield key
            yield from _string_tokens(child)
    elif isinstance(node, _Arr):
        for child in node.items:
            yield from _string_tokens(child)


def _walk_block(
    block: dict[str, Any], leaf: Leaf, tools: Mapping[str, str], context: str = ""
) -> dict[str, Any]:
    if _type_of(block) in _MEDIA or _type_of(block) in _UNTOUCHED:
        return block
    if block.get("type") == "tool_result":  # the answer of the tool named by its tool_use
        context = _field_context(tools.get(str(block.get("tool_use_id"))))
    out: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, str):
            out[key] = value if key in _STRUCTURAL else _walk_text(value, leaf, context)
        elif key in ("content", "summary") and isinstance(value, list):
            out[key] = _walk_content(value, leaf, tools, context)
        elif isinstance(value, dict | list | tuple):
            out[key] = _walk_data(value, leaf)
        else:
            out[key] = value
    return out


def _walk_content(content: Any, leaf: Leaf, tools: Mapping[str, str], context: str = "") -> Any:
    if isinstance(content, str):
        return _walk_text(content, leaf, context)
    if isinstance(content, list):
        return [
            _walk_text(item, leaf, context)
            if isinstance(item, str)
            else _walk_block(item, leaf, tools, context)
            if isinstance(item, dict)
            else item
            for item in content
        ]
    return content


def _tool_names(messages: Sequence[BaseMessage]) -> dict[str, str]:
    """tool call id -> tool name, from the tool calls and tool_use blocks in the history, so the
    result of a tool can be read with the tool's name in front ("get_date_of_birth": 12/03/1980)."""
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call.get("id"):
                names[str(call["id"])] = call["name"]
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    names[str(block["id"])] = str(block.get("name", ""))
    return names


def _walk_message(
    message: BaseMessage, leaf: Leaf, tools: Mapping[str, str] | None = None
) -> BaseMessage:
    """The message with `leaf` applied to every string the provider will read: the content (str or
    blocks), a tool call's arguments, and the older raw argument strings in additional_kwargs."""
    tools = tools or {}
    context = ""
    if isinstance(message, ToolMessage):  # a bare result is read with its tool's name in front
        context = _field_context(message.name or tools.get(message.tool_call_id))
    update: dict[str, Any] = {"content": _walk_content(message.content, leaf, tools, context)}
    if isinstance(message, AIMessage):
        if message.tool_calls:
            update["tool_calls"] = [
                {**call, "args": _walk_data(call["args"], leaf)} for call in message.tool_calls
            ]
        if message.invalid_tool_calls:
            update["invalid_tool_calls"] = [
                {**call, "args": _walk_raw(args, leaf)}
                if isinstance(args := call.get("args"), str)
                else call
                for call in message.invalid_tool_calls
            ]
        kwargs = message.additional_kwargs
        if "tool_calls" in kwargs or "function_call" in kwargs:
            new = dict(kwargs)
            if isinstance(kwargs.get("tool_calls"), list):
                new["tool_calls"] = [_walk_raw_call(call, leaf) for call in kwargs["tool_calls"]]
            if isinstance(kwargs.get("function_call"), dict):
                new["function_call"] = _walk_raw_function(kwargs["function_call"], leaf)
            update["additional_kwargs"] = new
    return message.model_copy(update=update)


def _walk_raw_function(function: dict[str, Any], leaf: Leaf) -> dict[str, Any]:
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        return {**function, "arguments": _walk_raw(arguments, leaf)}
    return function


def _walk_raw_call(call: Any, leaf: Leaf) -> Any:
    if isinstance(call, dict) and isinstance(call.get("function"), dict):
        return {**call, "function": _walk_raw_function(call["function"], leaf)}
    return call


def _media_count(content: Any) -> int:
    if isinstance(content, list):
        return sum(
            (1 if _type_of(item) in _MEDIA else _media_count(item.get("content")))
            for item in content
            if isinstance(item, dict)
        )
    return 0


class _Prepared:
    """A request after redaction: the messages to send, and what is needed to put the originals
    back. Made fresh for every call and never kept: the mapping is personal data."""

    def __init__(
        self,
        messages: list[BaseMessage],
        mapping: Mapping[str, str],
        counts: Mapping[str, int],
        unscanned: int,
    ) -> None:
        self.messages = messages
        self.mapping = mapping
        self.count = sum(counts.values())
        self.counts = counts
        self.unscanned = unscanned


class _NameFieldHoldsPersonalData(Exception):
    """A message's `name` matched a detector or a known value (only its type name is logged)."""


def prepare(redactor: Redactor, messages: Sequence[BaseMessage]) -> _Prepared:
    """Redact every string the provider will read, in one Redactor call so that numbering follows
    first appearance across the whole message list. Any failure raises RedactionFailed and nothing
    is sent."""
    failure: str | None = None
    try:
        strings: list[str] = []
        contexts: list[str] = []

        def collect(text: str, context: str) -> str:
            strings.append(text)
            contexts.append(context)
            return text

        tools = _tool_names(messages)
        for message in messages:
            _walk_message(message, collect, tools)
        names = [m.name for m in messages if m.name]
        if names and redactor.redact(names).texts != tuple(names):
            # A provider accepts only plain names there, so no placeholder fits: refuse.
            raise _NameFieldHoldsPersonalData
        result = redactor.redact(strings, contexts=contexts)
        replacements = iter(result.texts)
        rebuilt = [
            _walk_message(message, lambda _text, _context: next(replacements), tools)
            for message in messages
        ]
        if next(replacements, None) is not None:  # the two walks disagree: a bug, send nothing
            raise RuntimeError("the redacted texts did not match the message strings")
        unscanned = sum(_media_count(message.content) for message in messages)
        return _Prepared(rebuilt, result.mapping, result.counts, unscanned)
    except RedactionFailed:
        raise
    except Exception as exc:  # fail closed, and say only the type: the text may be in the error
        failure = type(exc).__name__
    log.error("llm_redaction_failed", error_type=failure)
    raise RedactionFailed


# --- putting the originals back ----------------------------------------------------------------


def _escaped(mapping: Mapping[str, str]) -> dict[str, str]:
    """The mapping with each original JSON-escaped, for text that is itself JSON: a quote, a
    backslash or a newline in an original must not break the document."""
    return {key: json.dumps(value, ensure_ascii=False)[1:-1] for key, value in mapping.items()}


_JSON_OPENING = re.compile(r"\s*(?:\{|\[\s*[^A-Za-z\s])")


def _opens_json(text: str) -> bool:
    """The rule for text that is JSON without the call having asked for a response format, the same
    for a whole answer and for the first text of a stream. It is JSON when it opens with `{`, or
    with `[` followed (after any spaces) by something that is not a letter. A placeholder is `[`
    and a letter, so an answer that starts with one is plain text. Missed on purpose: an array
    that starts with a letter (`[true, ...]`, `[null]`). A lone `[` is undecided (a stream waits
    for the next text). A misfire, rare and accepted: prose that opens with a footnote marker
    (`[1]`, `[2]`, `[*`) is taken as JSON, so a later original holding `"`, `\\` or a newline is
    shown escaped."""
    return _JSON_OPENING.match(text) is not None


class _Restorer:
    """Puts originals back into an answer. Made for one call; counts the placeholder-shaped
    strings it could not restore (invented or mangled by the model)."""

    def __init__(self, mapping: Mapping[str, str], *, json_text: bool = False) -> None:
        self.plain = mapping
        self.escaped = _escaped(mapping)
        self.unrestored = 0
        self._json_text = json_text

    def text(self, text: str) -> str:
        restored, missed = restore(text, self.plain)
        self.unrestored += missed
        return restored

    def raw(self, text: str) -> str:
        """A provider's transport string (an argument string, a fragment of one, JSON-mode text).
        Whole JSON is restored string by string, at each nesting level; anything else (a cut-off
        fragment) by substituting the JSON-escaped originals."""
        restored = self._restore_json(text)
        if restored is not None:
            return restored
        restored, missed = restore(text, self.escaped)
        self.unrestored += missed
        return restored

    def string(self, text: str) -> str:
        """The decoded text of a string value. JSON text inside it is restored at its own level, so
        an original with a quote or a backslash does not break the document; anything else is plain
        text."""
        if text.lstrip()[:1] in ("{", "["):
            restored = self._restore_json(text)
            if restored is not None:
                return restored
        return self.text(text)

    def _restore_json(self, text: str) -> str | None:
        """`text` as JSON with the placeholders in its strings and keys restored, each string at its
        own level (a string that is itself JSON text is restored inside); None if it is not a JSON
        object or array, or too deeply nested to read."""
        if text.lstrip()[:1] not in ("{", "["):
            return None
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict | list):
                return None
            edits: list[tuple[int, int, str]] = []
            for token in _string_tokens(_scan(text)):
                new = self.string(token.value)
                if new != token.value:
                    edits.append((token.start, token.end, json.dumps(new, ensure_ascii=False)))
        except (ValueError, RecursionError):
            return None
        pieces: list[str] = []
        position = 0
        for start, end, replacement in edits:
            pieces.append(text[position:start])
            pieces.append(replacement)
            position = end
        pieces.append(text[position:])
        return "".join(pieces)

    def answer(self, text: str) -> str:
        """Answer text: JSON text gets escaped originals. It is JSON when the call asked for JSON
        (_json_output) or the answer opens with { or [ (_opens_json), as in a stream."""
        return self.raw(text) if self._json_text or _opens_json(text) else self.text(text)

    def data(self, obj: Any) -> Any:
        """Plain data (tool arguments, a parsed answer): every string restored as text, JSON text
        inside a string at its own level. No field name changes how: a field called `arguments`
        in a tool's own schema is data like any other."""
        if isinstance(obj, BaseModel):
            # A parsed schema object (OpenAI's json_schema): copy it with restored fields. Not
            # validated again: the provider's SDK validated the placeholder form.
            fields = {name: self.data(getattr(obj, name)) for name in type(obj).model_fields}
            return obj.model_copy(update=fields)
        if isinstance(obj, str):
            return self.string(obj)
        if isinstance(obj, dict):
            return {
                # A key can hold a placeholder too (the model copied one into a field name).
                (self.string(key) if isinstance(key, str) else key): self.data(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list | tuple):
            return [self.data(value) for value in obj]
        return obj

    def block(self, block: dict[str, Any]) -> dict[str, Any]:
        """A content block. Thinking and reasoning blocks are left as they are. A block's own
        `arguments` or `partial_json` string is the provider's transport string (a Responses
        function_call, Anthropic's partial JSON), by its position in the block; the same name
        deeper down is a tool's own data."""
        kind = _type_of(block)
        if kind in _UNTOUCHED:
            return block
        if kind == "text" and isinstance(block.get("text"), str):
            return {**block, "text": self.answer(block["text"])}
        return {
            key: self.raw(value)
            if key in _RAW_KEYS and isinstance(value, str)
            else self.data(value)
            for key, value in block.items()
        }

    def content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self.answer(content)
        if not isinstance(content, list):
            return content
        return [
            self.answer(block)
            if isinstance(block, str)
            else self.block(block)
            if isinstance(block, dict)
            else block
            for block in content
        ]

    def transport_kwargs(self, key: str, value: Any) -> Any:
        """An additional_kwargs entry. The older raw argument strings of a tool call are transport
        strings; reasoning is left alone; everything else is data."""
        if key == "reasoning":
            return value
        if key == "tool_calls" and isinstance(value, list):
            return [self._transport_call(call) for call in value]
        if key == "function_call" and isinstance(value, dict):
            return self._transport_function(value)
        return self.data(value)

    def _transport_function(self, function: dict[str, Any]) -> dict[str, Any]:
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            return {**function, "arguments": self.raw(arguments)}
        return function

    def _transport_call(self, call: Any) -> Any:
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            return {**call, "function": self._transport_function(call["function"])}
        return call

    def message(self, message: AIMessage) -> AIMessage:
        update: dict[str, Any] = {"content": self.content(message.content)}
        if message.tool_calls:
            update["tool_calls"] = [
                {**call, "args": self.data(call["args"])} for call in message.tool_calls
            ]
        if message.invalid_tool_calls:
            update["invalid_tool_calls"] = [
                {**call, "args": self.raw(args)}
                if isinstance(args := call.get("args"), str)
                else call
                for call in message.invalid_tool_calls
            ]
        if message.additional_kwargs:
            update["additional_kwargs"] = {
                key: self.transport_kwargs(key, value)
                for key, value in message.additional_kwargs.items()
            }
        return message.model_copy(update=update)


def _json_output(kwargs: Mapping[str, Any]) -> bool:
    """Whether the call asks for JSON text as the answer (JSON mode, a schema), judged from the
    parameters bound to it: then the originals are put back JSON-escaped, so a quote in a value
    does not break the document. Tool calls are raw JSON anyway and always escaped."""
    if kwargs.get("response_format") or kwargs.get("output_format"):
        return True
    config = kwargs.get("output_config")
    if isinstance(config, dict) and config.get("format"):
        return True
    text = kwargs.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        return text["format"].get("type") not in (None, "text")
    return False


class _StreamRestorers:
    """Restores a stream of chunks for one call: a StreamRestorer for each stream of text that can
    be cut anywhere (a text block, a tool call's argument fragments), so a placeholder split
    across chunks is held back until it is whole. Made for one call."""

    def __init__(self, mapping: Mapping[str, str], *, json_text: bool) -> None:
        self._plain = mapping
        self._escaped = _escaped(mapping)
        self._json_text = json_text
        self._streams: dict[tuple[str, Any], StreamRestorer] = {}
        self._whole_missed = 0
        self._values = _Restorer(mapping)
        self._undecided: dict[Any, str] = {}

    def _stream(self, kind: str, index: Any, *, raw: bool) -> StreamRestorer:
        stream = self._streams.get((kind, index))
        if stream is None:
            stream = self._streams[(kind, index)] = StreamRestorer(
                self._escaped if raw else self._plain
            )
        return stream

    @property
    def unrestored(self) -> int:
        streams = sum(stream.unrestored for stream in self._streams.values())
        return self._whole_missed + self._values.unrestored + streams

    def _feed_text(self, index: Any, delta: str) -> str:
        """A text delta. Whether the stream is JSON text is decided at its first text that is not
        whitespace, by the rule in _opens_json; a lone `[` waits for the next text."""
        stream = self._streams.get(("text", index))
        if stream is not None:
            return stream.feed(delta)
        text = self._undecided.pop(index, "") + delta
        stripped = text.lstrip()
        if not stripped:
            return text
        if stripped == "[" or (stripped.startswith("[") and not stripped[1:].strip()):
            self._undecided[index] = text
            return ""
        raw = self._json_text or _opens_json(text)
        return self._stream("text", index, raw=raw).feed(text)

    def _kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """A chunk's additional_kwargs, restored the way `_Restorer.message` restores a whole
        answer's: complete values (a parsed schema object, structured-output fields) as they come;
        a refusal and the older raw tool-call arguments, which may arrive in fragments, through a
        restorer per stream; `reasoning` untouched."""
        out: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "reasoning":
                out[key] = value
            elif key == "refusal" and isinstance(value, str):
                out[key] = self._stream("refusal", None, raw=False).feed(value)
            elif key == "function_call" and isinstance(value, dict):
                arguments = value.get("arguments")
                feed = self._stream("kwargs-function", None, raw=True).feed
                out[key] = (
                    {**value, "arguments": feed(arguments)} if isinstance(arguments, str) else value
                )
            elif key == "tool_calls" and isinstance(value, list):
                out[key] = [self._raw_call(call) for call in value]
            else:
                out[key] = self._values.data(value)
        return out

    def _raw_call(self, call: Any) -> Any:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            feed = self._stream("kwargs-call", call.get("index"), raw=True).feed
            return {**call, "function": {**function, "arguments": feed(function["arguments"])}}
        return call

    def _whole(self, text: str) -> str:
        restored, missed = restore(text, self._plain)
        self._whole_missed += missed
        return restored

    def chunk(self, chunk: AIMessageChunk) -> AIMessageChunk:
        content = chunk.content
        if isinstance(content, str):
            content = self._feed_text(None, content)
        else:
            blocks: list[Any] = []
            for block in content:
                if not isinstance(block, dict):
                    blocks.append(block)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append(
                        {**block, "text": self._feed_text(block.get("index"), block["text"])}
                    )
                elif isinstance(block.get("partial_json"), str):
                    feed = self._stream("json", block.get("index"), raw=True).feed
                    blocks.append({**block, "partial_json": feed(block["partial_json"])})
                elif block.get("type") == "function_call" and isinstance(
                    block.get("arguments"), str
                ):
                    # The Responses API repeats each argument fragment here, beside
                    # tool_call_chunks: a thread keeps these blocks, so they are restored too.
                    feed = self._stream("args", block.get("index"), raw=True).feed
                    blocks.append({**block, "arguments": feed(block["arguments"])})
                elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                    # A complete refusal text, not a fragment.
                    blocks.append({**block, "refusal": self._whole(block["refusal"])})
                else:  # thinking, reasoning, images: as they are
                    blocks.append(block)
            content = blocks
        calls = [
            {**call, "args": self._stream("call", call.get("index"), raw=True).feed(args)}
            if isinstance(args := call.get("args"), str)
            else call
            for call in chunk.tool_call_chunks
        ]
        return AIMessageChunk(
            content=content,
            additional_kwargs=self._kwargs(chunk.additional_kwargs),
            response_metadata=chunk.response_metadata,
            id=chunk.id,
            name=chunk.name,
            tool_call_chunks=calls,
            usage_metadata=chunk.usage_metadata,
        )

    def flush(self) -> list[AIMessageChunk]:
        """Whatever was still held back at the end, as chunks (empty when nothing was)."""
        tails: list[AIMessageChunk] = []
        for index, text in self._undecided.items():  # a lone `[` that nothing followed: plain text
            stream = self._stream("text", index, raw=self._json_text)
            shown = stream.feed(text) + stream.flush()
            if index is None:
                tails.append(AIMessageChunk(content=shown))
            else:
                block = {"type": "text", "text": shown, "index": index}
                tails.append(AIMessageChunk(content=[block]))
        self._undecided.clear()
        for (kind, index), stream in self._streams.items():
            tail = stream.flush()
            if not tail:
                continue
            if kind == "text" and index is None:
                tails.append(AIMessageChunk(content=tail))
            elif kind == "text":
                tails.append(
                    AIMessageChunk(content=[{"type": "text", "text": tail, "index": index}])
                )
            elif kind == "refusal":
                tails.append(AIMessageChunk(content="", additional_kwargs={"refusal": tail}))
            elif kind == "kwargs-function":
                function = {"arguments": tail}
                tails.append(
                    AIMessageChunk(content="", additional_kwargs={"function_call": function})
                )
            elif kind == "kwargs-call":
                call = {"index": index, "function": {"arguments": tail}}
                tails.append(AIMessageChunk(content="", additional_kwargs={"tool_calls": [call]}))
            elif kind == "args":
                block = {"type": "function_call", "arguments": tail, "index": index}
                tails.append(AIMessageChunk(content=[block]))
            elif kind == "json":
                tails.append(
                    AIMessageChunk(
                        content=[{"type": "input_json_delta", "partial_json": tail, "index": index}]
                    )
                )
            else:
                tails.append(
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": None,
                                "id": None,
                                "args": tail,
                                "index": index,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
        return tails


def _repoint(runnable: Any, inner: BaseChatModel, outer: BaseChatModel, hits: list[int]) -> Any:
    """`runnable` with every use of `inner` replaced by `outer`, in the shapes LangChain's
    with_structured_output builds: a binding of the model, a sequence (model, then a parser), the
    include_raw map (raw model output beside a parsed one), and fallbacks. `hits` gets one entry
    per replacement."""
    if runnable is inner:
        hits.append(1)
        return outer
    if isinstance(runnable, RunnableBinding):
        if runnable.bound is inner:
            hits.append(1)
            return runnable.model_copy(update={"bound": outer})
        bound = _repoint(runnable.bound, inner, outer, hits)
        return runnable.model_copy(update={"bound": bound})
    if isinstance(runnable, RunnableSequence):
        steps = [_repoint(step, inner, outer, hits) for step in runnable.steps]
        return RunnableSequence(*steps, name=runnable.name)
    if isinstance(runnable, RunnableAssign):
        return RunnableAssign(_repoint(runnable.mapper, inner, outer, hits))
    if isinstance(runnable, RunnableParallel):
        mapped = {k: _repoint(v, inner, outer, hits) for k, v in runnable.steps__.items()}
        return RunnableParallel(mapped)
    if isinstance(runnable, RunnableWithFallbacks):
        return runnable.model_copy(
            update={
                "runnable": _repoint(runnable.runnable, inner, outer, hits),
                "fallbacks": [_repoint(f, inner, outer, hits) for f in runnable.fallbacks],
            }
        )
    return runnable


def _uses(runnable: Any, target: Any, depth: int = 0) -> bool:
    """Whether `target` is still reachable in a runnable built by LangChain (the fail-closed check
    after _repoint: any use left would send the original text to the provider)."""
    if runnable is target:
        return True
    if depth > 12:
        return True  # deeper than anything LangChain builds: do not guess
    children: list[Any] = []
    if isinstance(runnable, RunnableBinding):
        children = [runnable.bound]
    elif isinstance(runnable, RunnableSequence):
        children = list(runnable.steps)
    elif isinstance(runnable, RunnableAssign):
        children = [runnable.mapper]
    elif isinstance(runnable, RunnableParallel):
        children = list(runnable.steps__.values())
    elif isinstance(runnable, RunnableWithFallbacks):
        children = [runnable.runnable, *runnable.fallbacks]
    return any(_uses(child, target, depth + 1) for child in children)


# --- the wrapper -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallContext:
    """What the hidden inner run must inherit from the caller's call of the wrapper: the wrapper
    run's id (its parent), the tags and metadata (feature, user_ref and prompt among them, which
    the budget guard reads to apply the per-user cap), and the LangChainTracers."""

    parent_run_id: Any
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]
    tracers: tuple[Any, ...]


class _Holder:
    """Filled in when the wrapper's run starts, read by `_astream` when langchain-core calls it
    without a run manager (`astream`, and the streaming branch of `ainvoke`)."""

    def __init__(self, context: _CallContext | None = None) -> None:
        self.context = context


_pending: ContextVar[_Holder | None] = ContextVar("redaction_pending_call", default=None)


def _tracers(handlers: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(h for h in handlers if isinstance(h, LangChainTracer))


class _CaptureRun(AsyncCallbackHandler):
    """A handler on the wrapper's own run that records its id, tags and metadata, so `astream` can
    hand them to the inner run. It runs inline, in the caller's task, before `_astream` starts."""

    run_inline = True

    def __init__(self, holder: _Holder, tracers: tuple[Any, ...]) -> None:
        self._holder = holder
        self._tracers = tracers

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: Any,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._holder.context = _CallContext(
            run_id, tuple(tags or ()), dict(metadata or {}), self._tracers
        )


def _context_of(run_manager: AsyncCallbackManagerForLLMRun | None) -> _CallContext | None:
    if run_manager is None:
        return None
    return _CallContext(
        run_manager.run_id,
        tuple(run_manager.inheritable_tags),
        dict(run_manager.inheritable_metadata),
        _tracers(run_manager.inheritable_handlers),
    )


def _inner_config(context: _CallContext, count: int) -> RunnableConfig:
    """The config for the hidden inner call: the wrapper's tracers (so LangSmith shows the inner
    run under the wrapper's, masked), nothing else inherited (LangGraph's message stream would send
    the inner tokens to users), the wrapper's tags and metadata, `nostream`, and the redaction
    count."""
    handlers = list(context.tracers)
    manager = AsyncCallbackManager(
        handlers=handlers, inheritable_handlers=handlers, parent_run_id=context.parent_run_id
    )
    return cast(
        RunnableConfig,
        {
            "callbacks": manager,
            "tags": [*dict.fromkeys(context.tags), "nostream"],
            "metadata": {**context.metadata, "redacted_count": count},
        },
    )


def _missing_context() -> RedactionFailed:
    """Fail closed: without the caller's context the inner call would be reserved with no user,
    so the per-user cap would be skipped. Nothing is sent."""
    log.error("llm_redaction_failed", error_type="MissingCallContext")
    return RedactionFailed()


def _note(prepared: _Prepared) -> None:
    """Counts only, never values: the metric by kind, and a warning for media that was not
    scanned (images and other files are sent as they are)."""
    metrics.record_redactions(prepared.counts)
    if prepared.unscanned:
        log.warning("llm_unscanned_images", count=prepared.unscanned)


class RedactingChatModel(BaseChatModel):
    """A chat model that redacts personal data before the model inside it is called, and restores
    it in the answer. See the module notes."""

    _inner: BaseChatModel = PrivateAttr()
    _redactor: Redactor = PrivateAttr()

    def __init__(self, inner: BaseChatModel, redactor: Redactor) -> None:
        # cache=False: a LangChain cache would key on the originals; the inner model has its own.
        super().__init__(profile=inner.profile, cache=False)
        self._inner = inner
        self._redactor = redactor

    @property
    def inner(self) -> BaseChatModel:
        return self._inner

    # What create_agent, the step limit and LangSmith read from a model, forwarded from the
    # provider model so they behave as they did without the wrapper.
    @property
    def _llm_type(self) -> str:
        return self._inner._llm_type

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return dict(self._inner._identifying_params)

    @property
    def model_name(self) -> Any:
        return getattr(self._inner, "model_name", None)

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def use_responses_api(self) -> Any:
        return getattr(self._inner, "use_responses_api", None)

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return self._inner._get_ls_params(stop=stop, **kwargs)

    # Async only: LangChain swallows errors from async callbacks on the sync path (the budget
    # guard among them), so a sync call would skip the budget and the redaction check.
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError(
            "use the async API: await model.ainvoke(...) or model.astream(...)"
        )

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """The provider's own tool binding (its schemas, `strict`, `tool_choice`), pointed at this
        wrapper instead of the provider model, so tool calls are redacted too. Refuses to be built
        if the provider's binding is not a plain binding of its model."""
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        bound = self._inner.bind_tools(tools, **kwargs)
        if bound is self._inner:  # a model that takes no tools (a test double): nothing to bind
            return self
        if (
            isinstance(bound, RunnableBinding)
            and bound.bound is self._inner
            and not bound.config
            and not bound.config_factories
        ):
            return self.bind(**bound.kwargs)
        raise RedactionUnavailable(
            "bind_tools did not return a plain binding of the model, so its tool calls could not "
            "be redacted"
        )

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """The provider's own structured output (every method it has), with the model inside it
        replaced by this wrapper, so the request is redacted and the parsed fields are restored.
        Refuses to be built if the model is not found in what the provider returned, or is still
        reachable after the replacement."""
        built = self._inner.with_structured_output(schema, include_raw=include_raw, **kwargs)
        hits: list[int] = []
        repointed = _repoint(built, self._inner, self, hits)
        if not hits or _uses(repointed, self._inner):
            raise RedactionUnavailable(
                "with_structured_output did not bind the model in a shape this code knows, so "
                "its fields could not be redacted"
            )
        return cast("Runnable[Any, Any]", repointed)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        context = _context_of(run_manager) or self._pending_context()
        prepared = prepare(self._redactor, messages)
        _note(prepared)
        answer = await self._inner.ainvoke(
            prepared.messages,
            config=_inner_config(context, prepared.count),
            stop=stop,
            **kwargs,
        )
        if not isinstance(answer, AIMessage):
            raise TypeError("the model did not return an AIMessage")
        restorer = _Restorer(prepared.mapping, json_text=_json_output(kwargs))
        restored = restorer.message(answer)
        if restorer.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorer.unrestored)
        return ChatResult(generations=[ChatGeneration(message=restored)])

    @staticmethod
    def _pending_context() -> _CallContext:
        holder = _pending.get()
        _pending.set(None)  # read once: a later call must set its own
        if holder is None or holder.context is None:
            raise _missing_context()
        return holder.context

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """langchain-core calls `_astream` here without a run manager, so the call's tags, metadata
        and tracers are captured from the wrapper's own run and handed to `_astream` (a context
        variable holding a one-call holder). Without them the per-user cap would be skipped."""
        config = ensure_config(config)
        holder = _Holder()
        callbacks = config.get("callbacks")
        inheritable = (
            callbacks.inheritable_handlers
            if isinstance(callbacks, BaseCallbackManager)
            else list(callbacks or [])
        )
        capture = _CaptureRun(holder, _tracers(inheritable))
        if isinstance(callbacks, BaseCallbackManager):
            wrapped: Any = callbacks.copy()
            wrapped.add_handler(capture, inherit=False)
        else:
            wrapped = [*(callbacks or []), capture]
        _pending.set(holder)
        try:
            stream = cast(
                AsyncGenerator[AIMessageChunk, None],
                super().astream(input, {**config, "callbacks": wrapped}, stop=stop, **kwargs),
            )
            # aclosing: closing this stream closes langchain-core's, and with it the inner run.
            async with aclosing(stream) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            _pending.set(None)

    async def _agenerate_with_cache(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """The streaming branch of `ainvoke` (a streaming handler is attached, as LangGraph's
        message stream does) calls `_astream` without the run manager: hand it over here."""
        token = _pending.set(_Holder(_context_of(run_manager)))
        try:
            return await super()._agenerate_with_cache(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        finally:
            _pending.reset(token)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        context = _context_of(run_manager) or self._pending_context()
        prepared = prepare(self._redactor, messages)
        _note(prepared)
        restorers = _StreamRestorers(prepared.mapping, json_text=_json_output(kwargs))
        stream = cast(
            AsyncGenerator[AIMessageChunk, None],
            self._inner.astream(
                prepared.messages,
                config=_inner_config(context, prepared.count),
                stop=stop,
                **kwargs,
            ),
        )
        # aclosing: a consumer that stops early (or is cancelled) closes the inner stream too, so
        # the inner run ends and the budget guard settles or fails its reservation.
        async with aclosing(stream) as chunks:
            async for chunk in chunks:
                yield ChatGenerationChunk(message=restorers.chunk(chunk))
        for tail in restorers.flush():
            yield ChatGenerationChunk(message=tail)
        if restorers.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorers.unrestored)


_NEUTRAL = "[redaction failed]"


def _neutral(data: Any) -> dict[str, Any]:
    """What a failed mask returns: a key is text and can be personal data, so no key of the payload
    survives, at any depth. The one key LangSmith reads back is `error` (client.py `_hide_run_error`
    keeps a run's original error unless the result still has it), so that fixed key stays, with a
    neutral value, when the payload had it. Inputs, outputs and metadata are taken as they are,
    whatever keys they have, so an empty dict is a valid result for them."""
    return {"error": _NEUTRAL} if isinstance(data, dict) and "error" in data else {}


def masking_anonymizer(redactor: Redactor) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """For `langsmith.Client(anonymizer=...)`: every string in a run's inputs, outputs, metadata or
    error is masked before it is sent, one way (nothing is restored in a trace). It uses the
    redaction scope of the request that made the run, so the app's known values are masked too: the
    tracer's worker threads run in a copy of that request's context.

    If masking fails for any reason it returns nothing of the payload (see _neutral: not even its
    keys, which can be personal data; only the fixed `error` key LangSmith needs, neutralised), so
    the trace carries nothing rather than the text, and the failure is logged by type only."""

    def anonymize(data: dict[str, Any]) -> dict[str, Any]:
        failure: str | None = None
        try:
            strings: list[str] = []
            contexts: list[str] = []

            def collect(text: str, context: str) -> str:
                strings.append(text)
                contexts.append(context)
                return text

            _walk_data(data, collect)
            result = redactor.redact(strings, log_conflicts=False, contexts=contexts)
            replacements = iter(result.texts)
            masked = _walk_data(data, lambda _text, _context: next(replacements))
            if isinstance(masked, dict):
                return masked
        except RedactionFailed:  # already logged, by type only
            return _neutral(data)
        except Exception as exc:  # fail closed, and say only the type
            failure = type(exc).__name__
        log.error("llm_redaction_failed", error_type=failure or "TypeError", where="langsmith")
        return _neutral(data)

    return anonymize
