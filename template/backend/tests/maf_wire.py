"""A fake provider HTTP API under the real provider SDKs, for the MAF tests (#35).

The real OpenAI and Anthropic SDK clients run in full (the Responses and Messages request bodies
MAF builds, the SDK's retries, `retry-after` handling and timeouts) over an httpx2 MockTransport, so
a test sees the exact bytes that would go out and scripts what each HTTP attempt gets: a status, a
header, a delay, a dropped connection. No network, no key. MockTransport ignores timeouts, so the
fake enforces the attempt's read timeout itself.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2
import openai

__all__ = [
    "Attempt",
    "FakeProviderHttp",
    "Fault",
    "anthropic_message",
    "openai_response",
    "openai_sse",
    "sse",
]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One HTTP attempt as it arrived: the path, the JSON body, and when."""

    path: str
    body: dict[str, Any]
    at: float


@dataclass(frozen=True, slots=True)
class Fault:
    """What one attempt gets instead of a normal answer."""

    status: int = 500
    headers: dict[str, str] = field(default_factory=dict[str, str])
    # Seconds before answering; longer than the attempt's read timeout makes the SDK time out.
    delay: float = 0.0
    connection_error: bool = False


# (this attempt, its 1-based number) -> a fault, or None to answer normally
Plan = Callable[[Attempt, int], Fault | None]


def anthropic_message(text: str, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def openai_response(text: str, usage: dict[str, Any]) -> dict[str, Any]:
    """A Responses API answer (what `POST /responses` returns)."""
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


def _events(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def sse(
    text: str,
    usage: dict[str, int],
    *,
    pieces: list[str] | None = None,
    repeat_input_in_delta: bool = False,
) -> bytes:
    """An Anthropic streamed answer: message_start with the input (and cache) counts and one output
    token, the text in `pieces` (or whole), then message_delta with the CUMULATIVE usage, as
    Anthropic documents it ("The token counts shown in the usage field of the message_delta event
    are cumulative", platform.claude.com/docs/en/build-with-claude/streaming). With
    `repeat_input_in_delta` the delta repeats the input and cache counts too, as the docs' web
    search example does; a client that added the two up would count the input twice."""
    start = {**anthropic_message("", {**usage, "output_tokens": 1}), "content": []}
    delta_usage = (
        dict(usage) if repeat_input_in_delta else {"output_tokens": usage["output_tokens"]}
    )
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for piece in pieces or [text]:
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": delta_usage,
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return _events(events)


def openai_sse(text: str, usage: dict[str, Any], *, pieces: list[str] | None = None) -> bytes:
    """An OpenAI Responses streamed answer: the text deltas, then response.completed, the one
    event that carries the usage (developers.openai.com/api/reference/resources/responses/
    streaming-events)."""
    item = {"type": "message", "id": "msg_fake", "status": "in_progress", "role": "assistant"}
    created = {**openai_response("", usage), "status": "in_progress", "output": [], "usage": None}
    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"type": "response.created", "response": created}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "content": []},
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": "msg_fake",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ),
    ]
    for piece in pieces or [text]:
        events.append(
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_fake",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": piece,
                    "logprobs": [],
                },
            )
        )
    done_item = {
        **item,
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    events += [
        (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": "msg_fake",
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
        ),
        (
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": 0, "item": done_item},
        ),
        (
            "response.completed",
            {"type": "response.completed", "response": openai_response(text, usage)},
        ),
    ]
    numbered = [
        (name, {**data, "sequence_number": number}) for number, (name, data) in enumerate(events)
    ]
    return _events(numbered)


class FakeProviderHttp:
    """A fake `api.openai.com` or `api.anthropic.com`. `client` is the SDK's own httpx client over
    it: pass it to `Gateway(..., http_client=fake.client)`."""

    def __init__(
        self,
        provider: str,
        *,
        text: str = "4",
        usage: dict[str, Any] | None = None,
        plan: Plan | None = None,
        pieces: list[str] | None = None,
        repeat_input_in_delta: bool = False,
    ) -> None:
        assert provider in {"openai", "anthropic"}
        assert pieces is None or "".join(pieces) == text
        self.provider = provider
        self.text = text
        self.pieces = pieces
        self.repeat_input_in_delta = repeat_input_in_delta
        self.usage: dict[str, Any] = usage or (
            {"input_tokens": 11, "output_tokens": 3}
            if provider == "anthropic"
            else {
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        )
        self.plan: Plan = plan or (lambda _attempt, _number: None)
        self.attempts: list[Attempt] = []
        self.on_attempt: Callable[[Attempt], Awaitable[None]] | None = None
        transport = httpx2.MockTransport(self._handle)
        self.client: Any = (
            anthropic.DefaultAsyncHttpxClient(transport=transport)
            if provider == "anthropic"
            else openai.DefaultAsyncHttpxClient(transport=transport)
        )

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [attempt.body for attempt in self.attempts]

    async def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body: dict[str, Any] = json.loads(await request.aread() or b"{}")
        attempt = Attempt(path=request.url.path, body=body, at=time.monotonic())
        self.attempts.append(attempt)
        if self.on_attempt is not None:
            await self.on_attempt(attempt)
        fault = self.plan(attempt, len(self.attempts))
        if fault is not None:
            if fault.delay:
                # A real transport enforces the attempt's read timeout; MockTransport doesn't.
                timeouts: dict[str, float | None] = request.extensions.get("timeout", {})
                limit = timeouts.get("read")
                if limit is not None and fault.delay > limit:
                    await asyncio.sleep(limit)
                    raise httpx2.ReadTimeout("timed out", request=request)
                await asyncio.sleep(fault.delay)
            if fault.connection_error:
                raise httpx2.ConnectError("connection refused", request=request)
            error = {"type": "error", "error": {"type": "api_error", "message": "fake failure"}}
            # An empty x-should-retry leaves the decision to the SDK's own status rules.
            headers = {"x-should-retry": "", **fault.headers}
            return httpx2.Response(fault.status, headers=headers, json=error)
        streamed = {"content-type": "text/event-stream"}
        if self.provider == "anthropic":
            if body.get("stream"):
                content = sse(
                    self.text,
                    self.usage,
                    pieces=self.pieces,
                    repeat_input_in_delta=self.repeat_input_in_delta,
                )
                return httpx2.Response(200, headers=streamed, content=content)
            return httpx2.Response(200, json=anthropic_message(self.text, self.usage))
        if body.get("stream"):
            content = openai_sse(self.text, self.usage, pieces=self.pieces)
            return httpx2.Response(200, headers=streamed, content=content)
        return httpx2.Response(200, json=openai_response(self.text, self.usage))
