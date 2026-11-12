"""Personal-data redaction for MAF chat calls (#78, docs/decisions/0006).

`RedactionMiddleware` is the outer of the gateway's two chat middleware. Before a request goes down
the chain it replaces contact, bank and ID numbers and the values the app named (`redaction_scope`)
by placeholders such as [EMAIL_1] in every string the provider will read (the messages, a tool
call's arguments, a tool's result, the `instructions` option an agent sends its system prompt in,
and the tool definitions and output schema the call offers); after the answer comes back it puts the
originals back. The budget middleware inside it therefore sizes and records the redacted request,
and gets the number of values replaced through the call's `context.metadata`, a per-call dict, so
concurrent calls cannot mix their counts or mappings (nothing is stored on the middleware but the
redactor).

The walkers apply the core's rules (app/llm/core/redaction_data.py) to MAF's `Message` and
`Content`, and fail closed: a content type they do not know is not guessed at, and any error means
the call is refused with a 503 and nothing is sent (RedactionFailed; only the error's type name is
logged). The messages the caller holds are never changed: the walkers build copies, so a stored
thread keeps its originals.

What it does not do: find every kind of personal data, or anonymise (the mapping is the key, in
memory for one call). Thinking the provider signed is left as it is (it checks the signature), so
it can show placeholders; thinking sent without a signature is cleaned like text. Images and other
media are not scanned (counted in `llm_unscanned_images`; a document, audio or video is refused by
the budget). Options outside a fixed list (ALLOWED_OPTIONS in middleware.py) are refused, so
free-text ones (`user`, `metadata`, `extra_body`) never reach the provider. Chat middleware added
per call or per agent is refused by the gateway's clients (it would run after redaction and the
budget; see NoExtraChatMiddleware in gateway.py). A message's `author_name` cannot take a
placeholder, so a call whose name matches a detector or a known value is refused.

Streams: text is restored as it arrives (a placeholder split across updates is held back until it
is whole, by the core's StreamRestorer); a tool call's argument fragments are passed on empty and
the restored arguments come in one update per call at the end of the stream (`merge_mode`
"replace"), restored at each JSON nesting level as a whole answer's are (#102). The final response
is the provider's own, restored whole, so it reads exactly like the streamed text.

A call that asked for a schema (`response_format`) has its final answer checked here, after the
originals are back, whether redaction is on or off: an answer that does not parse into the schema
(or has no value, as a refusal) is the core's SchemaMismatch, a 502 settled at its cost. MAF would
otherwise parse it lazily, on the caller's first `.value`, outside the gateway. A response that
asks for tool calls is not an answer yet (a model call inside a tool loop) and is not checked.

Tool definitions and the output schema: a tool's description and parameter schema, and a
`response_format` (a mapping or a Pydantic model), are scanned like text, but a name, a JSON Schema
type, format or `$ref` is an identifier the provider checks and is left alone
(app/llm/core/redaction_data.py). A tool with nothing to replace is sent as the same object; one
with something to replace is sent as a copy carrying the redacted description and schema (the tool
the caller holds, and the function MAF runs, are not changed). A Pydantic `response_format` with
something to replace is sent as a subclass whose JSON schema is the redacted one, so each client
keeps its own conversion; the answer's value is parsed again into the caller's class. A schema rule
that holds personal data (`pattern`) is scanned like text and stops matching what it meant.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import structlog
from agent_framework import (
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionTool,
    Message,
    ResponseStream,
)

from app.llm.core import metrics
from app.llm.core.errors import schema_mismatch
from app.llm.core.redaction import RedactionFailed, Redactor, StreamRestorer
from app.llm.core.redaction_data import (
    Leaf,
    Restorer,
    escaped,
    field_context,
    opens_json,
    walk_data,
    walk_definition,
    walk_output_schema,
    walk_text,
)

__all__ = [
    "REDACTED_COUNT",
    "Prepared",
    "RedactionMiddleware",
    "RedactionUnavailable",
    "prepare",
]

log = structlog.get_logger()

# The key of the per-call metadata (ChatContext.metadata) that carries the number of values this
# call had replaced to the budget middleware: a count, never a value. Absent or None: off.
REDACTED_COUNT = "app.llm.redacted_count"

# Content types and what they hold. Anything else is refused (fail closed).
_TEXT = "text"
# Thinking the provider signed (`protected_data`) is never changed: it checks the signature. The
# Anthropic client sends thinking WITHOUT one as plain text, so that is walked like any text.
_REASONING = "text_reasoning"
_MEDIA = frozenset({"data", "uri", "hosted_file"})  # counted, sent as they are
_NOT_SENT = frozenset({"usage"})
_APPROVALS = frozenset({"function_approval_request", "function_approval_response"})
_PLAIN = (str, int, float, bool, type(None))


class RedactionUnavailable(RuntimeError):
    """The gateway cannot guarantee that every request it sends is redacted, so it refuses (fail
    closed): a client whose middleware chain lacks the redaction, or a result of a shape this code
    does not know."""


class _UnknownContent(Exception):
    """A content type or value the walkers do not know (only its type name is logged)."""


class _NameFieldHoldsPersonalData(Exception):
    """A message's `author_name` matched a detector or a known value."""


def _is_plain(value: Any) -> bool:
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    if isinstance(value, list | tuple):
        return all(_is_plain(v) for v in value)
    return False


def _walk_value(value: Any, leaf: Leaf, tools: Mapping[str, str], context: str) -> Any:
    """A tool result or a structured option: strings are redacted, Content inside it is walked,
    and a value of a type this code does not know is refused."""
    if isinstance(value, str):
        return walk_text(value, leaf, context)
    if isinstance(value, Content):
        return _walk_content(value, leaf, tools, context)
    if isinstance(value, _PLAIN):
        return value
    if _is_plain(value):
        return walk_data(value, leaf, context)
    if isinstance(value, list | tuple):
        return [_walk_value(item, leaf, tools, context) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _walk_value(item, leaf, tools, context) for key, item in value.items()}
    raise _UnknownContent(type(value).__name__)


def _walk_content(content: Content, leaf: Leaf, tools: Mapping[str, str], context: str = "") -> Any:
    """A copy of the content with `leaf` applied to every string the provider will read."""
    kind = content.type
    if kind in _MEDIA or kind in _NOT_SENT or (kind == _REASONING and content.protected_data):
        return content
    new = copy.copy(content)
    if kind in (_TEXT, _REASONING):
        new.text = walk_text(content.text or "", leaf, context)
    elif kind == "function_call":
        arguments = content.arguments
        if isinstance(arguments, str):
            new.arguments = walk_text(arguments, leaf, context)  # the raw JSON argument string
        elif arguments is not None:
            new.arguments = _walk_value(arguments, leaf, tools, context)
    elif kind == "function_result":
        # A result is read with its tool's name in front ("get_date_of_birth": 12/03/1980).
        where = field_context(tools.get(str(content.call_id)))
        new.result = _walk_value(content.result, leaf, tools, where)
        if content.items:
            new.items = [_walk_content(item, leaf, tools, where) for item in content.items]
    elif kind in _APPROVALS:
        if content.function_call is not None:
            new.function_call = _walk_content(content.function_call, leaf, tools, context)
    else:
        raise _UnknownContent(kind)
    return new


def _tool_names(messages: Sequence[Message]) -> dict[str, str]:
    """tool call id -> tool name, from the function calls in the history."""
    names: dict[str, str] = {}
    for message in messages:
        for content in message.contents:
            call = content.function_call if content.type in _APPROVALS else content
            if call is not None and call.type == "function_call" and call.call_id and call.name:
                names[str(call.call_id)] = str(call.name)
    return names


def _walk_message(message: Message, leaf: Leaf, tools: Mapping[str, str]) -> Message:
    return Message(
        message.role,
        [_walk_content(content, leaf, tools) for content in message.contents],
        author_name=message.author_name,
        message_id=message.message_id,
        additional_properties=message.additional_properties,
        raw_representation=message.raw_representation,
    )


def _walk_instructions(value: Any, leaf: Leaf) -> Any:
    if isinstance(value, str):
        return walk_text(value, leaf, "")
    if _is_plain(value):
        return walk_data(value, leaf, "")
    raise _UnknownContent(type(value).__name__)


def _walk_tool(tool: Any, leaf: Leaf) -> Any:
    """A tool as it is offered: a FunctionTool's description and parameter schema (the name is left
    alone), or a mapping MAF passes through as it is. Unchanged, the same object; changed, a copy.
    Anything else is refused: MAF would send it in a shape this code cannot see."""
    if isinstance(tool, FunctionTool):
        spec = {"description": tool.description or "", "parameters": tool.parameters()}
        walked = walk_definition(spec, leaf)
        if walked == spec:
            return tool
        new = copy.copy(tool)
        if tool.description is not None:
            new.description = walked["description"]
        # What parameters() returns from now on (agent-framework-core 1.25.0 caches it there; a
        # test pins the attribute, so a version that renames it fails the test, not a call).
        new._cached_parameters = walked["parameters"]
        return new
    if isinstance(tool, Mapping):
        return walk_output_schema(tool if isinstance(tool, dict) else dict(tool), leaf)
    raise _UnknownContent(type(tool).__name__)


def _walk_definitions(options: Mapping[str, Any], leaf: Leaf) -> dict[str, Any]:
    """The options with the tools and the output schema walked; unchanged values are the same
    objects."""
    out = dict(options)
    tools = options.get("tools")
    if tools is not None:
        if not isinstance(tools, list | tuple):
            raise _UnknownContent(type(tools).__name__)
        walked = [_walk_tool(tool, leaf) for tool in tools]
        if any(new is not old for new, old in zip(walked, tools, strict=True)):
            out["tools"] = walked
    if options.get("response_format") is not None:
        out["response_format"] = walk_output_schema(options["response_format"], leaf)
    return out


def _media_count(content: Content) -> int:
    if content.type in _MEDIA:
        return 1
    nested = [content.function_call] if content.function_call is not None else []
    nested += list(content.items or [])
    if isinstance(content.result, Content):
        nested.append(content.result)
    elif isinstance(content.result, list | tuple):
        nested += [item for item in content.result if isinstance(item, Content)]
    return sum(_media_count(item) for item in nested)


class Prepared:
    """A request after redaction: what to send, and what is needed to put the originals back. Made
    fresh for every call and never kept: the mapping is personal data."""

    def __init__(
        self,
        messages: list[Message],
        options: Mapping[str, Any],
        mapping: Mapping[str, str],
        counts: Mapping[str, int],
        unscanned: int,
    ) -> None:
        self.messages = messages
        self.options = options
        self.mapping = mapping
        self.count = sum(counts.values())
        self.counts = counts
        self.unscanned = unscanned


def prepare(
    redactor: Redactor, messages: Sequence[Message], options: Mapping[str, Any] | None = None
) -> Prepared:
    """Redact every string the provider will read, in one Redactor call so that numbering follows
    first appearance across the whole request (the tool definitions and output schema, then the
    instructions, then the messages: the order a provider reads them in, so a placeholder in a
    fixed definition keeps its number as a thread grows). Any failure raises RedactionFailed and
    nothing is sent."""
    failure: str | None = None
    try:
        options = options or {}
        strings: list[str] = []
        contexts: list[str] = []

        def collect(text: str, context: str) -> str:
            strings.append(text)
            contexts.append(context)
            return text

        tools = _tool_names(messages)
        _walk_definitions(options, collect)
        if options.get("instructions") is not None:
            _walk_instructions(options["instructions"], collect)
        for message in messages:
            _walk_message(message, collect, tools)
        names = [m.author_name for m in messages if m.author_name]
        if names and redactor.redact(names).texts != tuple(names):
            # A provider accepts only plain names there, so no placeholder fits: refuse.
            raise _NameFieldHoldsPersonalData
        result = redactor.redact(strings, contexts=contexts)
        replacements = iter(result.texts)

        def replace(_text: str, _context: str) -> str:
            return next(replacements)

        new_options = _walk_definitions(options, replace)
        if options.get("instructions") is not None:
            new_options["instructions"] = _walk_instructions(options["instructions"], replace)
        rebuilt = [_walk_message(message, replace, tools) for message in messages]
        if next(replacements, None) is not None:  # the two walks disagree: a bug, send nothing
            raise RuntimeError("the redacted texts did not match the message strings")
        unscanned = sum(_media_count(c) for message in messages for c in message.contents)
        return Prepared(rebuilt, new_options, result.mapping, result.counts, unscanned)
    except RedactionFailed:
        raise
    except Exception as exc:  # fail closed, and say only the type: the text may be in the error
        failure = type(exc).__name__
    log.error("llm_redaction_failed", error_type=failure)
    raise RedactionFailed


# --- putting the originals back ---------------------------------------------------------------


class _Restorer(Restorer):
    """Puts originals back into a MAF response, on top of the core's Restorer. Made for one call;
    counts the placeholder-shaped strings it could not restore (invented or mangled by the
    model)."""

    def content(self, content: Content) -> Content:
        kind = content.type
        if kind == _TEXT and isinstance(content.text, str):
            content.text = self.answer(content.text)
        elif kind == "function_call":
            if isinstance(content.arguments, str):
                content.arguments = self.raw(content.arguments)  # the provider's JSON string
            elif content.arguments is not None:
                content.arguments = self.data(content.arguments)
        elif kind == "function_result":
            content.result = self.data(content.result)
        # Thinking and reasoning, usage, media and the rest are left as they are.
        return content

    def response(self, response: ChatResponse[Any]) -> ChatResponse[Any]:
        for message in response.messages:
            for content in message.contents:
                self.content(content)
        return response


class _StreamRestorers:
    """Restores the updates of one streamed call (see the module notes). Made for one call."""

    def __init__(self, mapping: Mapping[str, str], *, json_text: bool) -> None:
        self._plain = mapping
        self._escaped = escaped(mapping)
        self._json_text = json_text
        self._text: dict[str | None, StreamRestorer] = {}
        self._undecided: dict[str | None, str] = {}
        self._held: dict[str, list[str]] = {}
        self._first: dict[str, Content] = {}
        self._last_call: str | None = None
        self._values = _Restorer(mapping, json_text=json_text)

    def _feed(self, key: str | None, delta: str) -> str:
        """A text delta. Whether this message's text is JSON is decided at its first text that is
        not whitespace (opens_json), as in LangChain's stream; a lone `[` waits for more."""
        stream = self._text.get(key)
        if stream is not None:
            return stream.feed(delta)
        text = self._undecided.pop(key, "") + delta
        stripped = text.lstrip()
        if not stripped:
            return text
        if stripped.startswith("[") and not stripped[1:].strip():
            self._undecided[key] = text
            return ""
        raw = self._json_text or opens_json(text)
        stream = self._text[key] = StreamRestorer(self._escaped if raw else self._plain)
        return stream.feed(text)

    def _replace(self, key: str | None, text: str) -> str:
        """A text that replaces everything streamed for its message so far (OpenAI's
        `response.output_text.done` arrives as one, `merge_mode` "replace"): restored whole, and
        what was held back for the message is dropped, since this text supersedes it."""
        self._text.pop(key, None)
        self._undecided.pop(key, None)
        return self._values.answer(text)

    def _hold(self, content: Content) -> Content:
        """A fragment of a tool call's argument string: kept, and passed on empty. A fragment that
        replaces the call's arguments starts them again."""
        call_id = content.call_id or self._last_call
        if not call_id:
            raise RedactionUnavailable("a streamed tool call fragment belongs to no call")
        self._last_call = call_id
        if content.merge_mode == "replace":
            self._held.pop(call_id, None)
        self._held.setdefault(call_id, []).append(str(content.arguments))
        self._first.setdefault(call_id, content)
        empty = copy.copy(content)
        empty.arguments = ""
        return empty

    def update(self, update: ChatResponseUpdate) -> ChatResponseUpdate:
        contents: list[Content] = []
        for content in update.contents:
            if content.type == _TEXT and isinstance(content.text, str):
                new = copy.copy(content)
                if content.merge_mode == "replace":
                    new.text = self._replace(update.message_id, content.text)
                else:
                    new.text = self._feed(update.message_id, content.text)
                contents.append(new)
            elif content.type == "function_call" and isinstance(content.arguments, str):
                contents.append(self._hold(content))
            elif content.type == "function_call" and content.arguments is not None:
                new = copy.copy(content)
                new.arguments = self._values.data(content.arguments)
                contents.append(new)
            else:  # thinking, usage, media, approvals: as they are
                contents.append(content)
        restored = copy.copy(update)
        restored.contents = contents
        return restored

    def tails(self) -> list[ChatResponseUpdate]:
        """What was held back, as updates: the end of each message's text, then each tool call's
        restored arguments, which replace the empty ones streamed so far."""
        out: list[ChatResponseUpdate] = []
        for key, text in self._undecided.items():  # a lone `[` nothing followed: plain text
            stream = self._text.setdefault(key, StreamRestorer(self._plain))
            shown = stream.feed(text)
            if shown:
                out.append(_text_update(key, shown))
        self._undecided.clear()
        for key, stream in self._text.items():
            tail = stream.flush()
            if tail:
                out.append(_text_update(key, tail))
        for call_id, fragments in self._held.items():
            whole = copy.copy(self._first[call_id])
            whole.arguments = self._values.raw("".join(fragments))
            whole.merge_mode = "replace"
            out.append(ChatResponseUpdate(role="assistant", contents=[whole]))
        self._held.clear()
        return out


def _text_update(message_id: str | None, text: str) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        role="assistant", contents=[Content.from_text(text)], message_id=message_id
    )


def _restored_stream(
    inner: object, prepared: Prepared, given: Mapping[str, Any]
) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
    """The budget middleware's stream with the originals put back in its updates and in its final
    response."""
    if not isinstance(inner, ResponseStream):
        raise RedactionUnavailable("a streamed call returned something other than a stream")
    source = cast("ResponseStream[ChatResponseUpdate, ChatResponse[Any]]", inner)
    json_text = _json_output(prepared.options)
    restorers = _StreamRestorers(prepared.mapping, json_text=json_text)

    async def updates() -> AsyncGenerator[ChatResponseUpdate]:
        try:
            async for update in source:
                yield restorers.update(update)
        finally:
            await source.close()
        for tail in restorers.tails():
            yield tail

    async def final(_updates: Sequence[ChatResponseUpdate]) -> ChatResponse[Any]:
        response = await source.get_final_response()
        restorer = _Restorer(prepared.mapping, json_text=json_text)
        restorer.response(response)
        _reparse(response, given)
        if restorer.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorer.unrestored)
        _check_structured(response, given)
        return response

    return ResponseStream(updates(), finalizer=final)


class RedactionMiddleware(ChatMiddleware):
    """Redacts the request and restores the answer of every chat call, streamed or not.
    `redactor` is None when LLM_REDACTION=off: nothing is changed and no count is recorded (the
    call's `redacted_count` stays NULL)."""

    def __init__(self, redactor: Redactor | None) -> None:
        self.redactor = redactor

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        redactor = self.redactor
        if redactor is None:
            context.metadata[REDACTED_COUNT] = None
            options = dict(context.options or {})
            await call_next()
            result = context.result
            if isinstance(result, ResponseStream):
                result.with_result_transform(lambda response: _parse_and_check(response, options))
            elif isinstance(result, ChatResponse):
                _parse_and_check(result, options)
            return
        given = context.options
        prepared = prepare(redactor, context.messages, given)
        metrics.record_redactions(prepared.counts)
        if prepared.unscanned:
            log.warning("llm_unscanned_images", count=prepared.unscanned)
        # The originals stay the source of truth for anything derived later (compaction): record
        # which message each redacted copy came from.
        for original, redacted in zip(context.messages, prepared.messages, strict=True):
            context.record_message_replacement(redacted, original)
        context.messages = prepared.messages
        context.options = prepared.options
        context.metadata[REDACTED_COUNT] = prepared.count
        await call_next()
        if context.stream:
            context.result = _restored_stream(context.result, prepared, given or {})
            return
        response = context.result
        if not isinstance(response, ChatResponse):
            raise RedactionUnavailable("the model call returned something other than a response")
        restorer = _Restorer(prepared.mapping, json_text=_json_output(prepared.options))
        restorer.response(response)
        _reparse(response, given or {})
        if restorer.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorer.unrestored)
        _check_structured(response, given or {})


def _parse_and_check(response: object, options: Mapping[str, Any]) -> None:
    """With redaction off: parse the answer into the caller's own `response_format` (as with it
    on, see _reparse), then check it."""
    if isinstance(response, ChatResponse):
        _reparse(response, options)
    _check_structured(response, options)


class _NoStructuredValue(Exception):
    """A final answer to a call that asked for a schema had no value to parse (a refusal, or no
    text)."""


def _check_structured(response: object, options: Mapping[str, Any]) -> None:
    """Raise the core's SchemaMismatch (502) when a final answer does not parse into the schema
    the call asked for (see the module notes). Returns None: as a stream's result transform it
    leaves the response as it is."""
    if options.get("response_format") is None or not isinstance(response, ChatResponse):
        return
    if any(c.type == "function_call" for message in response.messages for c in message.contents):
        return  # asks for tool calls: not an answer yet
    try:
        value = response.value
    except ValueError as error:  # pydantic's ValidationError is a ValueError, as is bad JSON
        raise schema_mismatch(error) from None
    if value is None:
        raise schema_mismatch(_NoStructuredValue()) from None


def _reparse(response: ChatResponse[Any], options: Mapping[str, Any]) -> None:
    """A structured answer's `value` is parsed from what the provider sent, which is the redacted
    text (OpenAI's parse path sets it, and MAF caches it): forget it, so it is parsed again from
    the restored text on first use, into the caller's own `response_format` (not the redacted
    subclass that was sent). These attributes are private to agent-framework-core 1.25.0;
    a test pins them, and a version bump that renames them fails it."""
    schema = options.get("response_format")
    if schema is not None:
        response._response_format = schema
        response._value = None
        response._value_parsed = False


def _json_output(options: Mapping[str, Any]) -> bool:
    """Whether the call asks for JSON text as the answer (a schema or JSON mode): then the
    originals are put back JSON-escaped, so a quote in a value does not break the document."""
    return bool(options.get("response_format"))
