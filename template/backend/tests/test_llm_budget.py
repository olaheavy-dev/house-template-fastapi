from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import date, timedelta
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import (
    ModelAPIError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.llm import prices
from app.llm.core import budget, report
from app.llm.core.budget import (
    LlmCall,
    LlmSpendDay,
    LlmSpendUserDay,
    expire_stale,
    reserve,
    reserve_for,
    settle,
)
from app.llm.langchain.callbacks import BudgetGuard

pytestmark = pytest.mark.anyio

MODEL = "anthropic:claude-sonnet-5-5"
PRICE = prices.PRICES[MODEL]
DAY = date(2026, 1, 2)
QUESTION = [HumanMessage("What is 2 + 2?")]


def answer(text: str, input_tokens: int = 1_000, output_tokens: int = 500) -> AIMessage:
    usage: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return AIMessage(text, usage_metadata=usage)


class FailingModel(GenericFakeChatModel):
    """Fails like a provider would: at once, or after streaming part of an answer."""

    stream_first: bool = False
    error: Exception = ModelTimeoutError("request timed out")

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise self.error

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.stream_first:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content="Fo"))
            if run_manager:
                await run_manager.on_llm_new_token("Fo", chunk=chunk)
            yield chunk
        raise self.error


class StreamingModel(GenericFakeChatModel):
    """Streams like the providers do: text first, usage in the last chunk."""

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        final = answer("")
        for piece in ("fo", "ur"):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=final.usage_metadata)
        )


@pytest.fixture
def guard(db_session: AsyncSession) -> Callable[..., BudgetGuard]:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def make(cap: int = 1_000_000) -> BudgetGuard:
        return BudgetGuard(
            model=MODEL,
            price=PRICE,
            cap_micro_usd=cap,
            max_tokens=500,
            sessionmaker=lambda: sessions,
            today=lambda: DAY,
        )

    return make


async def calls(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def committed(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def test_a_streamed_call_is_settled_from_its_final_usage(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = StreamingModel(messages=iter([]), callbacks=[guard()])
    text = "".join([chunk.text async for chunk in model.astream(QUESTION)])
    assert text == "four"
    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("settled", 7_000)


class ApiAndInvalid(ModelAPIError, ModelInvalidRequestError):
    pass


class RateLimitAndApi(ModelRateLimitError, ModelAPIError):
    pass


@pytest.mark.parametrize(
    ("error", "released"), [(ApiAndInvalid("x"), False), (RateLimitAndApi("x"), True)]
)
async def test_a_class_in_two_families_settles_by_its_first_matching_kind(
    guard: Callable[..., BudgetGuard],
    db_session: AsyncSession,
    error: Exception,
    released: bool,
) -> None:
    # Main released if ANY base was a rejected class; now the first matching row of
    # langchain/errors.py KINDS decides (safer: it never under-records spend).
    model = FailingModel(messages=iter([]), error=error, callbacks=[guard()])
    with pytest.raises(type(error)):
        await model.ainvoke(QUESTION)
    [call] = await calls(db_session)
    assert call.status == "failed"
    if released:
        assert (call.cost_micro_usd, await committed(db_session)) == (0, 0)
    else:
        assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_a_stream_that_breaks_midway_keeps_its_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    # The provider may bill what it generated before the break, and nothing reports how much.
    model = FailingModel(
        messages=iter([]),
        stream_first=True,
        error=ModelRateLimitError("even a rejection"),
        callbacks=[guard()],
    )
    with pytest.raises(ModelRateLimitError):
        _ = [chunk async for chunk in model.astream(QUESTION)]
    [call] = await calls(db_session)
    assert call.status == "failed"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_settling_twice_changes_nothing(db_session: AsyncSession) -> None:
    call_id = uuid.uuid4()
    assert await reserve(db_session, call_id=call_id, day=DAY, model=MODEL, amount=900, cap=1_000)
    await settle(db_session, call_id=call_id, status="settled", cost=100)
    await settle(db_session, call_id=call_id, status="settled", cost=50)
    assert await committed(db_session) == 100


async def user_total(session: AsyncSession, user_ref: str) -> int | None:
    return await session.scalar(
        select(LlmSpendUserDay.committed_micro_usd).where(
            LlmSpendUserDay.day == DAY, LlmSpendUserDay.user_ref == user_ref
        )
    )


async def test_a_refusal_by_the_day_cap_leaves_the_users_total_alone(
    db_session: AsyncSession,
) -> None:
    refused = await reserve_for(
        db_session,
        user_ref="u_3",
        user_cap=1_000_000,
        call_id=uuid.uuid4(),
        day=DAY,
        model=MODEL,
        amount=500,
        cap=100,
    )
    assert refused == "day"
    assert await user_total(db_session, "u_3") is None


async def test_a_call_that_never_reserved_against_the_user_leaves_their_total_alone(
    db_session: AsyncSession,
) -> None:
    # Codex review on PR 21: during a rollout an uncapped worker's call for the same user may
    # settle after a capped worker created the user's row; it must not subtract from it.
    capped, uncapped = uuid.uuid4(), uuid.uuid4()
    call = {"day": DAY, "model": MODEL, "amount": 900, "cap": 1_000_000}
    assert (
        await reserve_for(db_session, user_ref="u_4", user_cap=10_000, call_id=capped, **call)
        is None
    )
    assert (
        await reserve_for(db_session, user_ref="u_4", user_cap=None, call_id=uncapped, **call)
        is None
    )
    await settle(db_session, call_id=uncapped, status="settled", cost=100)
    assert await user_total(db_session, "u_4") == 900
    await settle(db_session, call_id=capped, status="settled", cost=100)
    assert await user_total(db_session, "u_4") == 100


async def test_stale_reservations_are_closed_at_their_reserved_cost(
    db_session: AsyncSession,
) -> None:
    stuck, fresh = uuid.uuid4(), uuid.uuid4()
    for call_id in (stuck, fresh):
        assert await reserve(
            db_session, call_id=call_id, day=DAY, model=MODEL, amount=500, cap=10**6
        )
    await db_session.execute(
        update(LlmCall)
        .where(LlmCall.id == stuck)
        .values(created_at=func.now() - timedelta(hours=2))
    )
    assert await report.stale(db_session) == 1
    assert await expire_stale(db_session, older_than=timedelta(minutes=30)) == 1
    closed = await db_session.get(LlmCall, stuck, populate_existing=True)
    assert closed is not None
    assert (closed.status, closed.cost_micro_usd) == ("failed", 500)
    assert await committed(db_session) == 1_000  # the day already counted both reservations
    assert await report.stale(db_session) == 0


async def test_the_report_counts_the_reservations_it_just_expired(
    db_session: AsyncSession,
) -> None:
    today_id = uuid.uuid4()
    assert await reserve(
        db_session, call_id=today_id, day=budget.utc_today(), model=MODEL, amount=700, cap=10**6
    )
    await db_session.execute(
        update(LlmCall)
        .where(LlmCall.id == today_id)
        .values(created_at=func.now() - timedelta(hours=1))
    )
    rows, stuck = await report.collect(db_session, days=1, by=["model"], expire=True)
    assert stuck == 0
    assert sum(row.cost_usd for row in rows) == pytest.approx(700 / prices.MICRO_USD)


@pytest.fixture
def concurrent_day(database_url: str) -> Iterator[date]:
    """A day no other test uses, cleaned up afterwards: these reservations commit for real."""
    day = date(2000, 1, 1)
    yield day

    async def clean() -> None:
        engine = create_async_engine(database_url, poolclass=NullPool)
        async with engine.begin() as connection:
            await connection.execute(delete(LlmCall).where(LlmCall.day == day))
            await connection.execute(delete(LlmSpendDay).where(LlmSpendDay.day == day))
        await engine.dispose()

    asyncio.run(clean())


async def test_concurrent_reservations_never_overshoot_the_cap(
    database_url: str, concurrent_day: date
) -> None:
    engine = create_async_engine(database_url, pool_size=10)
    sessions = async_sessionmaker(engine)

    async def one() -> bool:
        async with sessions() as session, session.begin():
            return await reserve(
                session,
                call_id=uuid.uuid4(),
                day=concurrent_day,
                model=MODEL,
                amount=300,
                cap=1_000,
            )

    results = await asyncio.gather(*(one() for _ in range(10)))
    async with sessions() as session:
        total = await session.scalar(
            select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == concurrent_day)
        )
    await engine.dispose()
    assert results.count(True) == 3
    assert total == 900


async def test_the_budget_has_its_own_pool(app: FastAPI) -> None:
    # A request holding a session from the main pool must never wait on that pool to reserve.
    async with LifespanManager(app):
        budget = app.state.budget_sessionmaker.kw["bind"]
        assert budget is not app.state.engine
        assert budget.pool.size() == 2
