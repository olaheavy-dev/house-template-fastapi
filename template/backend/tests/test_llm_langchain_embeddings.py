"""GatewayEmbeddings, the LangChain-facing embeddings class (#33 T4), through its public calls with
the provider client faked and the database real. The first tests pin what the class did before
embeddings moved onto the core's Meter; the later ones pin what the move changes on purpose. Money
is worked out by hand in the comments, never by the code under test: EMBED costs 20,000 micro-USD
per million tokens, which is 0.02 micro-USD per token, and a cost rounds UP to a whole micro-USD.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import BudgetExhausted, LlmCall, LlmSpendDay
from app.llm.core.prices import Price, Rates
from app.llm.embeddings import GatewayEmbeddings

pytestmark = pytest.mark.anyio

MODEL = "openai:pin-embeddings"
DAY = date(2026, 1, 3)
EMBED = Price(Rates(20_000, 0, 0, 0, 0), checked="2026-11-06", source="https://example.com/pricing")


class FakeEmbeddingsApi:
    """Answers like OpenAI's embeddings endpoint: one vector per input, 7 tokens per input."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[list[str]] = []
        self.error = error

    async def create(self, *, model: str, input: list[str], **kwargs: Any) -> Any:
        self.requests.append(list(input))
        if self.error:
            raise self.error
        data = [SimpleNamespace(index=i, embedding=[float(i)] * 3) for i in range(len(input))]
        return SimpleNamespace(
            data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=7 * len(input))
        )


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def embeddings(
    sessions: async_sessionmaker[AsyncSession],
) -> Callable[..., tuple[GatewayEmbeddings, FakeEmbeddingsApi]]:
    def make(
        cls: type[GatewayEmbeddings] = GatewayEmbeddings,
        cap: int = 10**9,
        price: Price = EMBED,
        client: Any = None,
    ) -> tuple[GatewayEmbeddings, FakeEmbeddingsApi]:
        fake = FakeEmbeddingsApi()
        return (
            cls(
                model=MODEL,
                price=price,
                client=client or SimpleNamespace(embeddings=fake),
                cap_micro_usd=cap,
                sessionmaker=lambda: sessions,
                today=lambda: DAY,
            ),
            fake,
        )

    return make


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def total(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def test_no_texts_send_no_request_and_leave_no_row(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    e, fake = embeddings()
    assert await e.aembed_documents([]) == []
    assert fake.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


async def test_a_batch_the_budget_cant_cover_is_refused_after_the_earlier_ones_settled(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    class OneAtATime(GatewayEmbeddings):
        MAX_INPUTS = 1

    e, fake = embeddings(OneAtATime, cap=21)
    with pytest.raises(BudgetExhausted):
        await e.aembed_documents(["a", "b"])
    # Batch 1: ("a" 1 byte + 1,024 margin) = 1,025 tokens x 0.02 = 20.5 -> reserved 21 (<= cap 21).
    # It settles at 7 tokens x 0.02 = 0.14 -> 1, so the day holds 1. Batch 2 would reserve 21 more:
    # 1 + 21 = 22 > 21, refused, and never sent.
    assert fake.requests == [["a"]]
    [row] = await rows(db_session)
    assert (row.status, row.reserved_micro_usd, row.cost_micro_usd) == ("settled", 21, 1)
    assert await total(db_session) == 1


async def test_the_cap_is_read_on_every_call_so_changing_it_applies_at_once(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    e, fake = embeddings()
    e.cap = 0
    with pytest.raises(BudgetExhausted):
        await e.aembed_documents(["a"])
    assert fake.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


async def test_the_feature_and_user_are_recorded_with_the_call_and_its_latency_is_measured(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    e, _ = embeddings()
    await e.aembed_documents(["a"], feature="ingest", user_ref="u_1")
    [row] = await rows(db_session)
    assert (row.feature, row.user_ref, row.prompt, row.prompt_sha) == ("ingest", "u_1", None, None)
    assert row.latency_ms is not None
