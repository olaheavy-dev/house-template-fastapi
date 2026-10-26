from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import date
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import ModelConnectionError, ModelRateLimitError, ModelTimeoutError
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.llm import budget, prices
from app.llm.budget import (
    BudgetExhausted,
    BudgetGuard,
    LlmCall,
    LlmSpendDay,
    UnboundedMedia,
    reserve,
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
    with pytest.raises(UnboundedMedia, match="'file'"):
        await model.ainvoke([HumanMessage(content=[pdf])])
    assert next(sent).text == "4"
    assert await calls(db_session) == []


async def test_settling_twice_changes_nothing(db_session: AsyncSession) -> None:
    call_id = uuid.uuid4()
    assert await reserve(db_session, call_id=call_id, day=DAY, model=MODEL, amount=900, cap=1_000)
    await settle(db_session, call_id=call_id, status="settled", cost=100)
    await settle(db_session, call_id=call_id, status="settled", cost=50)
    assert await committed(db_session) == 100


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
