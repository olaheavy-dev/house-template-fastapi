"""A fake provider HTTP API under the real provider SDKs, for the MAF tests (adapted from the
founder's MAF learning lab, #35).

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

__all__ = ["Attempt", "FakeProviderHttp", "Fault", "anthropic_message", "openai_response", "sse"]


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


def sse(text: str, usage: dict[str, int]) -> bytes:
    """An Anthropic streamed answer with its usage in message_start and message_delta (the shape
    MAF's streaming merge reads; used from the streaming PR on)."""
    start = {**anthropic_message("", {**usage, "output_tokens": 1}), "content": []}
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
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


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
    ) -> None:
        assert provider in {"openai", "anthropic"}
        self.provider = provider
        self.text = text
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
        if self.provider == "anthropic":
            if body.get("stream"):
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse(self.text, self.usage),
                )
            return httpx2.Response(200, json=anthropic_message(self.text, self.usage))
        return httpx2.Response(200, json=openai_response(self.text, self.usage))
