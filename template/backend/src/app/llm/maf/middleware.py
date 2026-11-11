"""The daily budget as MAF chat middleware: BudgetMiddleware reads what MAF reports about a call
(its messages and options, the response's usage, the error) and drives the core's Meter
(app/llm/core/budget.py), which reserves, settles and records the call.

It is the inner of the gateway's two chat middleware (RedactionMiddleware is outside it), so it
sees the redacted request. The client's layer order is FunctionInvocation, ChatMiddleware,
ChatTelemetry, the provider: ChatMiddleware runs for every model call, also each one inside an
agent's tool loop, so each call has its own reservation and its own `llm_call` row.

What it does to every call, in this order:
- refuses what it cannot bound or price: a document, audio or video, a tool the provider runs or a
  dearer service tier (core/bounds.py), and `store=True` (history stays in our database: the
  provider is told `store=False` on every call, which OpenAI's Responses API would otherwise
  default to True);
- sets `max_tokens` from LLM_MAX_TOKENS when the call sets none (Anthropic's client would silently
  use 1,024), and reserves for the larger of that and a limit the call set itself;
- reserves the most the call can cost, then calls, then settles from the response's usage (or
  keeps the reservation when the provider reported none), or records the failure: a refusal the
  provider answered before doing any work releases the reservation, anything else keeps it, and a
  cancelled call is left reserved until `expire_stale` closes it.

Streamed calls are not supported yet and are refused.

The tags of a call (the prompt, the feature, whom it is for) reach the middleware through context
variables, because MAF has no per-call metadata path that is verified for 1.25.0: use
`Gateway.call_options` (app/llm/maf/gateway.py), which sets them for the calls made inside its
`with` block, tool loops included.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog
from agent_framework import (
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    Content,
    FunctionTool,
    Message,
)

from app.llm.core import bounds
from app.llm.core.budget import CallTags, Meter
from app.llm.core.usage import Usage
from app.llm.maf.errors import classify
from app.llm.maf.redaction import REDACTED_COUNT, StreamingNotSupported
from app.llm.maf.usage import from_maf

__all__ = [
    "BudgetMiddleware",
    "StoreNotAllowed",
    "current_tags",
    "params_view",
    "request_view",
    "tagged",
]

log = structlog.get_logger()

_tags: ContextVar[CallTags | None] = ContextVar("llm_call_tags", default=None)
_MEDIA = frozenset({"data", "uri", "hosted_file"})


class StoreNotAllowed(ValueError):
    """A call asked the provider to store the conversation (`store=True`). The gateway keeps
    history in our own database and sends `store=False`; a call that says otherwise is a mistake in
    the code that made it."""


@contextmanager
def tagged(tags: CallTags) -> Iterator[None]:
    """The tags every call made inside the block (in this task and the tasks it starts) carries."""
    token = _tags.set(tags)
    try:
        yield
    finally:
        _tags.reset(token)


def current_tags() -> CallTags:
    return _tags.get() or CallTags()


def _tool_view(tool: Any) -> Any:
    """A tool as the budget's check wants to see it: a dict with a `type` for a tool the provider
    runs (a mapping MAF passes through), `function` for a tool the app runs, and an unknown object
    as its class name, which check_priced refuses (an object it cannot recognise is not assumed to
    be free)."""
    if isinstance(tool, Mapping):
        return dict(tool)
    if isinstance(tool, FunctionTool) or callable(tool):
        return {"type": "function", "name": getattr(tool, "name", None)}
    return {"type": type(tool).__name__}


def params_view(options: Mapping[str, Any]) -> dict[str, Any]:
    """The call's options as the neutral parameters bounds.check_priced and bounds.output_bound
    read: `tools`, `service_tier`, `max_tokens` and `max_completion_tokens`."""
    view: dict[str, Any] = {
        key: options[key]
        for key in ("service_tier", "max_tokens", "max_completion_tokens")
        if key in options
    }
    tools = options.get("tools")
    if tools is not None:
        view["tools"] = [_tool_view(tool) for tool in tools]
    return view


def _content_view(content: Content) -> dict[str, Any]:
    try:
        return content.to_dict()
    except Exception:  # an exotic content must not stop the call from being sized
        return {"type": content.type, "text": str(content.text), "repr": repr(content)}


def _blocks(content: Content) -> list[dict[str, Any]]:
    """The media in one content as the budget counts them: `image` for an image, `file` for any
    other media (which count_images refuses: a document, audio and video have no per-item bound)."""
    if content.type in _MEDIA:
        media_type = content.media_type or ""
        return [{"type": "image" if media_type.startswith("image/") else "file"}]
    nested: list[Content] = []
    if content.function_call is not None:
        nested.append(content.function_call)
    nested += list(content.items or [])
    if isinstance(content.result, Content):
        nested.append(content.result)
    elif isinstance(content.result, list | tuple):
        nested += [item for item in content.result if isinstance(item, Content)]
    return [block for item in nested for block in _blocks(item)]


def request_view(
    messages: Sequence[Message], options: Mapping[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """(payload, media blocks) for bounds.request_bound: everything sent (the messages, the tools,
    the instructions, the schema) as JSON-able data, and the media in it, which count at a fixed
    size each however they are sent."""
    payload = [
        [
            {
                "role": message.role,
                "name": message.author_name,
                "contents": [_content_view(content) for content in message.contents],
            }
            for message in messages
        ],
        {key: value for key, value in options.items() if key != "tools"},
        params_view(options).get("tools"),
    ]
    blocks = [block for message in messages for c in message.contents for block in _blocks(c)]
    return payload, blocks


class BudgetMiddleware(ChatMiddleware):
    """Reserves before each call of the client it is attached to and settles after it, through its
    Meter. `max_tokens` is the gateway's output limit (LLM_MAX_TOKENS)."""

    def __init__(self, meter: Meter, *, max_tokens: int, sends_store: bool) -> None:
        self.meter = meter
        self.max_tokens = max_tokens
        # Whether the provider's API has a `store` option to send False in: OpenAI's Responses API
        # does (and stores by default); Anthropic's Messages API stores nothing and its SDK refuses
        # the keyword.
        self.sends_store = sends_store

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if context.stream:
            raise StreamingNotSupported("streamed calls are not supported by this gateway yet")
        options = dict(context.options or {})
        if options.get("store") is True:
            raise StoreNotAllowed(
                "store=True would keep the conversation at the provider; "
                "the gateway sends store=False and keeps history in the database"
            )
        view = params_view(options)
        bounds.check_priced(view)
        # A caller may set a larger limit than the gateway's; reserve for whichever is larger.
        output = bounds.output_bound(self.max_tokens, view)
        if not isinstance(options.get("max_tokens"), int):
            options["max_tokens"] = self.max_tokens
        if self.sends_store:
            options["store"] = False
        else:
            options.pop("store", None)
        payload, blocks = request_view(context.messages, options)
        given = bounds.request_bound(payload, blocks)
        counted = context.metadata.get(REDACTED_COUNT)
        tags = current_tags()
        call_id = uuid.uuid4()
        await self.meter.reserve(
            call_id,
            input_tokens=given,
            output_tokens=output,
            tags=CallTags(
                prompt=tags.prompt,
                prompt_sha=tags.prompt_sha,
                feature=tags.feature,
                user_ref=tags.user_ref,
                # Set by the redaction middleware (a count, never a value); absent when it is off.
                redacted_count=counted if type(counted) is int else None,
            ),
        )
        context.options = options
        try:
            await call_next()
        except Exception as error:
            known = classify(error)
            await self.meter.failed(call_id, known.kind if known else None, streamed=False)
            if known is None:
                raise
            raise known from error
        except BaseException:
            # Cancelled: the call may still be running at the provider, so its row stays reserved
            # until expire_stale closes it. Nothing is awaited here, which would be cancelled again.
            self.meter.abandon(call_id)
            raise
        await self.meter.settle(call_id, _usage(context.result))


def _usage(result: object) -> Usage | None:
    if not isinstance(result, ChatResponse):
        return None
    return from_maf(result.usage_details)
