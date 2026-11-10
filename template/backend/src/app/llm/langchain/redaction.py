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
- Images and other media are not scanned (they are counted in `llm_unscanned_images`), and neither
  is a message's `name` field (a provider only accepts plain names there, so no placeholder fits).
- A LangChainTracer you pass in a call's config with a client of its own records what that client
  is given; only the gateway's client masks.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing
from typing import Any, cast

import structlog
from langchain_core.callbacks import AsyncCallbackManager, AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from pydantic import BaseModel, PrivateAttr

from app.llm.core.redaction import RedactionFailed, Redactor, StreamRestorer, restore

__all__ = ["RedactingChatModel", "RedactionUnavailable"]

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

Leaf = Callable[[str], str]


class RedactionUnavailable(RuntimeError):
    """The wrapper cannot guarantee that every request it builds is redacted, so it refuses to be
    built (fail closed). Raised when a LangChain method returns a shape this code does not
    know."""


# --- walking the strings of a message ----------------------------------------------------------


def _walk_data(obj: Any, leaf: Leaf) -> Any:
    """Every string value in plain data (tool arguments, a parsed answer), keys untouched."""
    if isinstance(obj, str):
        return leaf(obj)
    if isinstance(obj, dict):
        return {key: _walk_data(value, leaf) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_walk_data(value, leaf) for value in obj]
    return obj


def _walk_raw(text: str, leaf: Leaf) -> str:
    """Raw JSON text (an argument string): its string values are walked, and the text is written
    again only when one changed, so an untouched string is sent byte for byte. Text that is not
    JSON is one string."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return leaf(text)
    if not isinstance(parsed, dict | list):
        return leaf(text)
    changed = False

    def track(value: str) -> str:
        nonlocal changed
        out = leaf(value)
        changed = changed or out != value
        return out

    rebuilt = _walk_data(parsed, track)
    return json.dumps(rebuilt, ensure_ascii=False, separators=(",", ":")) if changed else text


def _walk_block(block: dict[str, Any], leaf: Leaf) -> dict[str, Any]:
    if block.get("type") in _MEDIA or block.get("type") in _UNTOUCHED:
        return block
    out: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, str):
            if key in _STRUCTURAL:
                out[key] = value
            elif key in _RAW_KEYS:
                out[key] = _walk_raw(value, leaf)
            else:
                out[key] = leaf(value)
        elif key in ("content", "summary") and isinstance(value, list):
            out[key] = _walk_content(value, leaf)
        elif isinstance(value, dict | list | tuple):
            out[key] = _walk_data(value, leaf)
        else:
            out[key] = value
    return out


def _walk_content(content: Any, leaf: Leaf) -> Any:
    if isinstance(content, str):
        return leaf(content)
    if isinstance(content, list):
        return [
            leaf(item)
            if isinstance(item, str)
            else _walk_block(item, leaf)
            if isinstance(item, dict)
            else item
            for item in content
        ]
    return content


def _walk_message(message: BaseMessage, leaf: Leaf) -> BaseMessage:
    """The message with `leaf` applied to every string the provider will read: the content (str or
    blocks), a tool call's arguments, and the older raw argument strings in additional_kwargs."""
    update: dict[str, Any] = {"content": _walk_content(message.content, leaf)}
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
            (1 if item.get("type") in _MEDIA else _media_count(item.get("content")))
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


def prepare(redactor: Redactor, messages: Sequence[BaseMessage]) -> _Prepared:
    """Redact every string the provider will read, in one Redactor call so that numbering follows
    first appearance across the whole message list. Any failure raises RedactionFailed and nothing
    is sent."""
    failure: str | None = None
    try:
        strings: list[str] = []

        def collect(text: str) -> str:
            strings.append(text)
            return text

        for message in messages:
            _walk_message(message, collect)
        result = redactor.redact(strings)
        replacements = iter(result.texts)
        rebuilt = [_walk_message(message, lambda _text: next(replacements)) for message in messages]
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


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


class _Restorer:
    """Puts originals back into an answer. Made for one call; counts the placeholder-shaped
    strings it could not restore (invented or mangled by the model)."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.plain = mapping
        self.escaped = _escaped(mapping)
        self.unrestored = 0

    def text(self, text: str) -> str:
        restored, missed = restore(text, self.plain)
        self.unrestored += missed
        return restored

    def raw(self, text: str) -> str:
        restored, missed = restore(text, self.escaped)
        self.unrestored += missed
        return restored

    def answer(self, text: str) -> str:
        """Answer text: JSON text (JSON mode, a schema answer) gets escaped originals."""
        return self.raw(text) if _looks_like_json(text) else self.text(text)

    def data(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            # A parsed schema object (OpenAI's json_schema): copy it with restored fields. Not
            # validated again: the provider's SDK validated the placeholder form.
            fields = {name: self.data(getattr(obj, name)) for name in type(obj).model_fields}
            return obj.model_copy(update=fields)
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            if obj.get("type") in _UNTOUCHED:
                return obj
            return {
                key: self.raw(value)
                if key in _RAW_KEYS and isinstance(value, str)
                else self.data(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list | tuple):
            return [self.data(value) for value in obj]
        return obj

    def content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self.answer(content)
        if not isinstance(content, list):
            return content
        out: list[Any] = []
        for block in content:
            if isinstance(block, str):
                out.append(self.answer(block))
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append({**block, "text": self.answer(str(block.get("text", "")))})
            else:
                out.append(self.data(block))
        return out

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
                key: value if key == "reasoning" else self.data(value)
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
    chosen = kwargs.get("ls_structured_output_format")
    if isinstance(chosen, dict) and isinstance(chosen.get("kwargs"), dict):
        return chosen["kwargs"].get("method") in ("json_mode", "json_schema")
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

    def _stream(self, kind: str, index: Any, *, raw: bool) -> StreamRestorer:
        stream = self._streams.get((kind, index))
        if stream is None:
            stream = self._streams[(kind, index)] = StreamRestorer(
                self._escaped if raw else self._plain
            )
        return stream

    @property
    def unrestored(self) -> int:
        return sum(stream.unrestored for stream in self._streams.values())

    def chunk(self, chunk: AIMessageChunk) -> AIMessageChunk:
        content = chunk.content
        if isinstance(content, str):
            content = self._stream("text", None, raw=self._json_text).feed(content)
        else:
            blocks: list[Any] = []
            for block in content:
                if not isinstance(block, dict):
                    blocks.append(block)
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    feed = self._stream("text", block.get("index"), raw=self._json_text).feed
                    blocks.append({**block, "text": feed(block["text"])})
                elif isinstance(block.get("partial_json"), str):
                    feed = self._stream("json", block.get("index"), raw=True).feed
                    blocks.append({**block, "partial_json": feed(block["partial_json"])})
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
            additional_kwargs=chunk.additional_kwargs,
            response_metadata=chunk.response_metadata,
            id=chunk.id,
            name=chunk.name,
            tool_call_chunks=calls,
            usage_metadata=chunk.usage_metadata,
        )

    def flush(self) -> list[AIMessageChunk]:
        """Whatever was still held back at the end, as chunks (empty when nothing was)."""
        tails: list[AIMessageChunk] = []
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


# --- the wrapper -------------------------------------------------------------------------------


def _inner_config(run_manager: AsyncCallbackManagerForLLMRun | None, count: int) -> RunnableConfig:
    """The config for the hidden inner call: the wrapper's tracers (so LangSmith shows the inner
    run under the wrapper's, masked), nothing else inherited (LangGraph's message stream would send
    the inner tokens to users), the wrapper's tags and metadata, `nostream`, and the redaction
    count."""
    handlers: list[Any] = []
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    parent = None
    if run_manager is not None:
        handlers = [h for h in run_manager.inheritable_handlers if isinstance(h, LangChainTracer)]
        tags = list(run_manager.inheritable_tags)
        metadata = dict(run_manager.inheritable_metadata)
        parent = run_manager.run_id
    manager = AsyncCallbackManager(
        handlers=handlers, inheritable_handlers=handlers, parent_run_id=parent
    )
    return cast(
        RunnableConfig,
        {
            "callbacks": manager,
            "tags": [*tags, "nostream"],
            "metadata": {**metadata, "redacted_count": count},
        },
    )


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

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prepared = prepare(self._redactor, messages)
        answer = await self._inner.ainvoke(
            prepared.messages,
            config=_inner_config(run_manager, prepared.count),
            stop=stop,
            **kwargs,
        )
        if not isinstance(answer, AIMessage):
            raise TypeError("the model did not return an AIMessage")
        restorer = _Restorer(prepared.mapping)
        restored = restorer.message(answer)
        if restorer.unrestored:
            log.warning("llm_unrestored_placeholders", count=restorer.unrestored)
        return ChatResult(generations=[ChatGeneration(message=restored)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prepared = prepare(self._redactor, messages)
        restorers = _StreamRestorers(prepared.mapping, json_text=_json_output(kwargs))
        stream = cast(
            AsyncGenerator[AIMessageChunk, None],
            self._inner.astream(
                prepared.messages,
                config=_inner_config(run_manager, prepared.count),
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
