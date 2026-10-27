from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import date, timedelta
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import ModelConnectionError, ModelRateLimitError, ModelTimeoutError
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.llm import budget, prices, report
from app.llm.budget import (
    BudgetExhausted,
    BudgetGuard,
    LlmCall,
    LlmSpendDay,
    LlmSpendUserDay,
    PromptTooLong,
    UnboundedMedia,
    UnpricedRequest,
    UserBudgetExhausted,
    expire_stale,
    reserve,
    reserve_for,
    settle,
)

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


async def test_a_call_is_settled_at_its_real_cost(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    assert (await model.ainvoke(QUESTION)).text == "4"

    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("settled", 7_000)  # 1,000 x $2/M + 500 x $10/M
    assert (call.input_tokens, call.output_tokens) == (1_000, 500)
    assert call.reserved_micro_usd >= (call.cost_micro_usd or 0)
    assert await committed(db_session) == 7_000


async def test_prompt_metadata_is_recorded_with_the_call(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    await model.ainvoke(QUESTION, config={"metadata": {"prompt": "sums", "prompt_sha": "abc"}})
    [call] = await calls(db_session)
    assert (call.prompt, call.prompt_sha) == ("sums", "abc")


async def test_a_call_that_could_cross_the_cap_is_never_sent(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    sent = iter([answer("4")])
    model = GenericFakeChatModel(messages=sent, callbacks=[guard(cap=1_000)])
    with pytest.raises(BudgetExhausted):
        await model.ainvoke(QUESTION)
    assert next(sent).text == "4"  # the model never answered
    assert await calls(db_session) == []
    assert await committed(db_session) is None


async def test_a_streamed_call_is_settled_from_its_final_usage(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = StreamingModel(messages=iter([]), callbacks=[guard()])
    text = "".join([chunk.text async for chunk in model.astream(QUESTION)])
    assert text == "four"
    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("settled", 7_000)


async def test_a_call_the_provider_rejected_releases_its_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = FailingModel(
        messages=iter([]), error=ModelRateLimitError("slow down"), callbacks=[guard()]
    )
    with pytest.raises(ModelRateLimitError):
        await model.ainvoke(QUESTION)
    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("failed", 0)
    assert await committed(db_session) == 0


@pytest.mark.parametrize(
    "error", [ModelTimeoutError("timed out"), ModelConnectionError("reset"), RuntimeError("?")]
)
async def test_a_call_that_may_have_been_processed_keeps_its_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession, error: Exception
) -> None:
    # A timeout or a dropped connection can come after the provider generated (and billed).
    model = FailingModel(messages=iter([]), error=error, callbacks=[guard()])
    with pytest.raises(type(error)):
        await model.ainvoke(QUESTION)
    [call] = await calls(db_session)
    assert call.status == "failed"
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


async def test_a_call_without_usage_keeps_its_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("4")]), callbacks=[guard()])
    await model.ainvoke(QUESTION)
    [call] = await calls(db_session)
    assert call.status == "settled"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_a_larger_bound_limit_is_reserved_for(guard: Callable[..., BudgetGuard]) -> None:
    # 500 output tokens fit a $0.01 cap; a caller binding 5,000 must not slip through.
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard(cap=10_000)])
    with pytest.raises(BudgetExhausted):
        await model.bind(max_tokens=5_000).ainvoke(QUESTION)


async def test_images_are_reserved_at_their_maximum_however_they_are_sent(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    image = {"type": "image", "url": "https://example.com/chart.png"}
    model = GenericFakeChatModel(messages=iter([answer("4"), answer("4")]), callbacks=[guard()])
    await model.ainvoke(QUESTION)
    await model.ainvoke([HumanMessage(content=[image, {"type": "text", "text": "What is 2 + 2?"}])])
    text_only, with_image = sorted(c.reserved_micro_usd for c in await calls(db_session))
    # At least IMAGE_TOKENS more input at the highest input rate ($4/M for 1h cache writes).
    assert with_image - text_only >= budget.IMAGE_TOKENS * 4


async def test_media_the_budget_cant_bound_is_refused_before_sending(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    pdf = {"type": "file", "url": "https://example.com/report.pdf", "mime_type": "application/pdf"}
    sent = iter([answer("4")])
    model = GenericFakeChatModel(messages=sent, callbacks=[guard()])
    with pytest.raises(UnboundedMedia) as refused:
        await model.ainvoke([HumanMessage(content=[pdf])])
    assert (refused.value.status, refused.value.kind) == (422, "file")
    assert next(sent).text == "4"
    assert await calls(db_session) == []


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


async def test_one_users_cap_refuses_only_that_user(
    db_session: AsyncSession, guard: Callable[..., BudgetGuard]
) -> None:
    capped = guard()
    capped.user_cap = 1_000_000
    as_user: RunnableConfig = {"metadata": {"user_ref": "u_1"}}
    first = GenericFakeChatModel(messages=iter([answer("4"), answer("4")]), callbacks=[capped])
    await first.ainvoke(QUESTION, config=as_user)
    # The first call settled at 7,000; a cap just short of 7,000 plus a fresh reservation
    # leaves no room for a second call.
    reservation = await db_session.scalar(select(LlmCall.reserved_micro_usd))
    assert reservation is not None
    capped.user_cap = 7_000 + reservation - 1
    with pytest.raises(UserBudgetExhausted) as refused:
        await first.ainvoke(QUESTION, config=as_user)
    assert (refused.value.status, int(refused.value.headers["Retry-After"]) > 0) == (429, True)
    # Someone else is unaffected, and the refusal left the day's total as it was.
    before = await committed(db_session)
    other = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[capped])
    await other.ainvoke(QUESTION, config={"metadata": {"user_ref": "u_2"}})
    assert await user_total(db_session, "u_1") == 7_000
    assert await user_total(db_session, "u_2") == 7_000
    assert await committed(db_session) == (before or 0) + 7_000


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


async def test_a_prompt_past_the_priced_length_is_refused_before_reserving(
    db_session: AsyncSession, guard: Callable[..., BudgetGuard]
) -> None:
    short_limit = guard()
    short_limit.price = prices.Price(PRICE.rates, max_prompt_tokens=10)
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[short_limit])
    with pytest.raises(PromptTooLong) as refused:
        await model.ainvoke(QUESTION)
    assert refused.value.status == 422
    assert await calls(db_session) == []


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


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search"},  # OpenAI, billed per search
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},  # Anthropic
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
)
async def test_a_tool_the_provider_runs_and_bills_per_use_is_refused_before_reserving(
    db_session: AsyncSession, guard: Callable[..., BudgetGuard], tool: dict[str, Any]
) -> None:
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    with pytest.raises(UnpricedRequest, match=str(tool["type"])):
        await model.bind(tools=[tool]).ainvoke(QUESTION)
    assert await calls(db_session) == []


async def test_tools_the_app_runs_cost_only_tokens_and_are_allowed(
    db_session: AsyncSession, guard: Callable[..., BudgetGuard]
) -> None:
    openai_style = {"type": "function", "function": {"name": "f", "parameters": {}}}
    anthropic_style = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    await model.bind(tools=[openai_style, anthropic_style]).ainvoke(QUESTION)
    assert [c.status for c in await calls(db_session)] == ["settled"]


@pytest.mark.parametrize("tier", ["priority", "scale"])
async def test_a_service_tier_priced_above_the_standard_rates_is_refused(
    db_session: AsyncSession, guard: Callable[..., BudgetGuard], tier: str
) -> None:
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    with pytest.raises(UnpricedRequest, match=tier):
        await model.bind(service_tier=tier).ainvoke(QUESTION)
    assert await calls(db_session) == []


@pytest.mark.parametrize("tier", ["auto", "default", "flex", "standard_only"])
async def test_service_tiers_at_or_below_the_standard_rates_are_allowed(
    guard: Callable[..., BudgetGuard], tier: str
) -> None:
    model = GenericFakeChatModel(messages=iter([answer("4")]), callbacks=[guard()])
    await model.bind(service_tier=tier).ainvoke(QUESTION)
