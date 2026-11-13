"""Streamed MAF calls (#35 PR 3): how a stream ends decides how its budget row ends, and what a
streamed tool call carries. The shared rules (settled from final usage, stopped early, rejected,
refused by the cap, restored text) are in the parity tests; these are the MAF-only cases. The
values are synthetic."""

from __future__ import annotations

import asyncio
import gc
import json
from datetime import date
from typing import Any

import pytest
from agent_framework import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    UsageDetails,
    tool,
)
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import LlmCall
from app.llm.core.errors import ProviderError, SchemaMismatch
from app.llm.maf.gateway import Gateway
from app.settings import Settings
from tests.harness import Failure, Reply
from tests.harness.maf import HARNESS, ScriptedClient, error_of

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 8)
EMAIL = "priya.shah@example.com"


@pytest.fixture
def gateway(settings: Settings, db_session: AsyncSession) -> Gateway:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    built = Gateway(
        settings.model_copy(update={"llm_model": "anthropic:claude-sonnet-5-5"}), lambda: sessions
    )
    built.meter.cap, built.meter.today = 1_000_000, lambda: DAY
    return built


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(await session.scalars(select(LlmCall).where(LlmCall.day == DAY)))


def script(gateway: Gateway, *items: Any) -> ScriptedClient:
    HARNESS.install_scripted(gateway, list(items))
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    return client


async def test_a_stream_dropped_without_closing_still_settles(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    # The documented way to stop early is `async with` (a bare break "doesn't provide
    # deterministic cleanup", learn.microsoft.com .../agents/running-agents). The gateway does not
    # rely on it: its own async generator is closed by the event loop's finalizer once the stream
    # is garbage, so the row still ends.
    script(gateway, Reply("one two three", pieces=("one ", "two ", "three")))

    async def read_one() -> None:
        stream = gateway.client().get_response([Message("user", ["hi"])], stream=True)
        async for _update in stream:
            break  # no close, no `async with`

    await read_one()
    gc.collect()
    for _ in range(20):  # the finalizer's aclose runs as a task on the loop
        await asyncio.sleep(0.01)
        if (await rows(db_session))[0].status != "reserved":
            break
    [row] = await rows(db_session)
    assert row.status == "failed"
    assert row.cost_micro_usd == row.reserved_micro_usd


async def test_a_cancelled_stream_is_left_reserved(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    # As for an unstreamed call: the provider may still be generating, so nothing is settled and
    # expire_stale closes the row later.
    gate = asyncio.Event()
    script(gateway, Reply("one two", pieces=("one ", "two")))
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    real = client._stream

    def slow(messages: Any, options: Any) -> ResponseStream[Any, Any]:
        inner = real(messages, options)

        async def updates() -> Any:
            async for update in inner:
                yield update
                await asyncio.sleep(10)  # the provider is still generating

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    client._stream = slow  # type: ignore[method-assign]

    async def consume() -> None:
        stream = gateway.client().get_response([Message("user", ["hi"])], stream=True)
        async with stream:
            async for _update in stream:
                gate.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(gate.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [row] = await rows(db_session)
    assert row.status == "reserved"


async def test_a_failure_after_output_keeps_the_reservation_and_is_answered_like_any_failure(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    client = script(gateway, Reply("x"))

    def breaks(messages: Any, options: Any) -> ResponseStream[Any, Any]:
        async def updates() -> Any:
            yield ChatResponseUpdate(role="assistant", contents=[Content.from_text("par")])
            raise error_of(Failure("rate_limit"))

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    client._stream = breaks  # type: ignore[method-assign]
    with pytest.raises(ProviderError):
        await HARNESS.stream(gateway, [{"role": "user", "content": "hi"}])
    [row] = await rows(db_session)
    # A rate limit before any output releases the reservation (parity tests); after output the
    # provider may already bill, so it is kept.
    assert row.status == "failed"
    assert row.cost_micro_usd == row.reserved_micro_usd


async def test_a_streamed_tool_call_shows_no_placeholder_and_the_tool_gets_the_original(
    gateway: Gateway,
) -> None:
    called: list[str] = []

    @tool
    def send_to(address: str) -> str:
        """Send the summary."""
        called.append(address)
        return "sent"

    # The arguments arrive in fragments that split the placeholder.
    fragments = ['{"address": "[EM', 'AIL_1]"}']
    asks = ChatResponse(
        messages=[
            Message(
                "assistant",
                [
                    Content.from_function_call("c1", "send_to", arguments=piece)
                    for piece in fragments
                ],
            )
        ]
    )
    client = script(gateway, asks, Reply("done"))
    shown: list[Any] = []
    stream = gateway.client().get_response(
        [Message("user", [f"Send it to {EMAIL}"])], stream=True, options={"tools": [send_to]}
    )
    async with stream:
        async for update in stream:
            shown += [c.arguments for c in update.contents if c.type == "function_call"]
        final = await stream.get_final_response()
    assert called == [EMAIL]
    assert not any("[EM" in str(arguments) or "AIL_1" in str(arguments) for arguments in shown)
    # The whole restored arguments arrived once, at the end of the model call's stream.
    assert {"address": EMAIL} in [json.loads(a) for a in shown if isinstance(a, str) and a]
    assert final.text == "done"
    assert EMAIL not in str(client.seen[0])


class Lead(BaseModel):
    email: str


async def test_a_streamed_structured_answer_is_the_callers_class_with_the_originals(
    gateway: Gateway,
) -> None:
    pieces = ('{"email": "[EM', 'AIL_1]"}')
    script(gateway, Reply("".join(pieces), pieces=pieces))
    stream = gateway.client().get_response(
        [Message("user", [f"Extract {EMAIL}"])], stream=True, options={"response_format": Lead}
    )
    async with stream:
        final = await stream.get_final_response()
    assert type(final.value) is Lead
    assert final.value.email == EMAIL


async def test_a_streamed_answer_that_does_not_fit_its_schema_is_the_502(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    script(gateway, Reply('{"email": 5}', pieces=('{"email": ', "5}")))
    stream = gateway.client().get_response(
        [Message("user", ["Extract the lead"])], stream=True, options={"response_format": Lead}
    )
    with pytest.raises(SchemaMismatch) as raised:
        async with stream:
            await stream.get_final_response()
    # Nothing of the answer travels with it: no cause, and its message is the fixed detail.
    assert raised.value.__cause__ is None
    assert "5" not in str(raised.value)
    [row] = await rows(db_session)
    assert row.status == "settled"  # the provider did the work


async def test_a_model_call_that_asks_for_tools_is_not_checked_against_the_schema(
    gateway: Gateway,
) -> None:
    # Inside a tool loop the first model call only asks for the tool: it has no structured value
    # yet and is not a mismatch; the final answer is checked.
    @tool
    def find(name: str) -> str:
        """Find a lead."""
        return "found"

    asks = ChatResponse(
        messages=[Message("assistant", [Content.from_function_call("c1", "find", arguments="{}")])]
    )
    script(gateway, asks, Reply('{"email": "x@y.z"}'))
    response = await gateway.client().get_response(
        [Message("user", ["Find the lead"])], options={"tools": [find], "response_format": Lead}
    )
    assert response.value == Lead(email="x@y.z")


async def test_usage_before_a_rejection_is_not_output_and_the_reservation_is_released(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    # Anthropic sends its usage in message_start before any text: a rejection after only that
    # still did no work, so it releases like an unstreamed rejection.
    client = script(gateway, Reply("x"))

    def usage_then_rejected(messages: Any, options: Any) -> ResponseStream[Any, Any]:
        async def updates() -> Any:
            usage = UsageDetails(input_token_count=11, output_token_count=1)
            yield ChatResponseUpdate(role="assistant", contents=[Content.from_usage(usage)])
            raise error_of(Failure("rate_limit"))

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    client._stream = usage_then_rejected  # type: ignore[method-assign]
    with pytest.raises(ProviderError):
        await HARNESS.stream(gateway, [{"role": "user", "content": "hi"}])
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("failed", 0)


async def test_stopping_early_closes_the_providers_stream(gateway: Gateway) -> None:
    closed: list[bool] = []
    client = script(gateway, Reply("one two", pieces=("one ", "two")))
    real = client._stream

    def recorded(messages: Any, options: Any) -> ResponseStream[Any, Any]:
        return real(messages, options).with_cleanup_hook(lambda: closed.append(True))

    client._stream = recorded  # type: ignore[method-assign]
    assert await HARNESS.stream(gateway, [{"role": "user", "content": "hi"}], take=1) == ["one "]
    assert closed == [True]


async def test_an_empty_answer_to_a_call_that_asked_for_a_schema_is_the_502(
    gateway: Gateway,
) -> None:
    # A refusal or an empty answer has no value to parse: as much a mismatch as bad JSON.
    script(gateway, Reply(""))
    with pytest.raises(SchemaMismatch) as raised:
        await gateway.client().get_response(
            [Message("user", ["Extract the lead"])], options={"response_format": Lead}
        )
    assert raised.value.__cause__ is None
