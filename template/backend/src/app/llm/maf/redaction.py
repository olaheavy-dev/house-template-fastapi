"""Personal-data redaction for MAF chat calls (#78, docs/decisions/0006).

`RedactionMiddleware` is the outer of the gateway's two chat middleware. Before a request goes
down the chain it replaces contact, bank and ID numbers and the values the app named
(`redaction_scope`) by placeholders such as [EMAIL_1] in every string the provider will read (the
messages, a tool call's arguments, a tool's result, and the `instructions` option an agent sends
its system prompt in); after the answer comes back it puts the originals back. The budget
middleware inside it therefore sizes and records the redacted request, and gets the number of
values replaced through the call's `context.metadata`, a per-call dict, so concurrent calls cannot
mix their counts or mappings (nothing is stored on the middleware but the redactor).

The walkers apply the core's rules (app/llm/core/redaction_data.py) to MAF's `Message` and
`Content`, and fail closed: a content type they do not know is not guessed at, and any error means
the call is refused with a 503 and nothing is sent (RedactionFailed; only the error's type name is
logged). The messages the caller holds are never changed: the walkers build copies, so a stored
thread keeps its originals.

What it does not do: find every kind of personal data, or anonymise (the mapping is the key, in
memory for one call). Thinking and reasoning content is left as it is (a provider checks its
signature), so it can show placeholders. Images and other media are not scanned (counted in
`llm_unscanned_images`; a document, audio or video is refused by the budget). Other options that
are sent as text (`user`, `metadata`) are not scanned: never put personal data there. An agent's
own chat middleware, added per run, is appended after this client-level chain, so it runs closer to
the provider than redaction and sees placeholders. A message's `author_name` cannot take a
placeholder, so a call whose name matches a detector or a known value is refused. Streamed calls
are not supported yet and are refused.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import structlog
from agent_framework import ChatContext, ChatMiddleware, ChatResponse, Content, Message

from app.llm.core import metrics
from app.llm.core.redaction import RedactionFailed, Redactor
from app.llm.core.redaction_data import (
    Leaf,
    Restorer,
    field_context,
    walk_data,
    walk_text,
)

__all__ = [
    "REDACTED_COUNT",
    "Prepared",
    "RedactionMiddleware",
    "RedactionUnavailable",
    "StreamingNotSupported",
    "prepare",
]

log = structlog.get_logger()

# The key of the per-call metadata (ChatContext.metadata) that carries the number of values this
# call had replaced to the budget middleware: a count, never a value. Absent or None: off.
REDACTED_COUNT = "app.llm.redacted_count"

# Content types and what they hold. Anything else is refused (fail closed).
_TEXT = "text"
_UNTOUCHED = frozenset({"text_reasoning"})  # a provider checks its signature: never changed
_MEDIA = frozenset({"data", "uri", "hosted_file"})  # counted, sent as they are
_NOT_SENT = frozenset({"usage"})
_APPROVALS = frozenset({"function_approval_request", "function_approval_response"})
_PLAIN = (str, int, float, bool, type(None))


class RedactionUnavailable(RuntimeError):
    """The gateway cannot guarantee that every request it sends is redacted, so it refuses (fail
    closed): a client whose middleware chain lacks the redaction, or a result of a shape this code
    does not know."""


class StreamingNotSupported(RuntimeError):
    """A streamed call reached the gateway's middleware. Streams are not supported yet: a stream
    would show the caller placeholders and its usage would not be settled."""


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
    if kind in _UNTOUCHED or kind in _MEDIA or kind in _NOT_SENT:
        return content
    new = copy.copy(content)
    if kind == _TEXT:
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
    first appearance across the whole request (the instructions, then the messages). Any failure
    raises RedactionFailed and nothing is sent."""
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

        new_options = dict(options)
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


class RedactionMiddleware(ChatMiddleware):
    """Redacts the request and restores the answer of every non-streamed chat call. `redactor` is
    None when LLM_REDACTION=off: nothing is changed and no count is recorded (the call's
    `redacted_count` stays NULL)."""

    def __init__(self, redactor: Redactor | None) -> None:
        self.redactor = redactor

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if context.stream:
            raise StreamingNotSupported("streamed calls are not supported by this gateway yet")
        redactor = self.redactor
        if redactor is None:
            context.metadata[REDACTED_COUNT] = None
            await call_next()
            return
        prepared = prepare(redactor, context.messages, context.options)
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
        response = context.result
        if not isinstance(response, ChatResponse):
            raise RedactionUnavailable("the model call returned something other than a response")
        restorer = _Restorer(prepared.mapping, json_text=_json_output(prepared.options))
        restorer.response(response)
        if restorer.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorer.unrestored)


def _json_output(options: Mapping[str, Any]) -> bool:
    """Whether the call asks for JSON text as the answer (a schema or JSON mode): then the
    originals are put back JSON-escaped, so a quote in a value does not break the document."""
    return bool(options.get("response_format"))
