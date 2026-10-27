"""Embeddings go through the same daily budget as chat calls: reserved before, settled from the
provider's usage after. The provider client is faked; the database is real."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from langchain_core.exceptions import ModelInvalidRequestError, ModelTimeoutError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import prices
from app.llm.budget import BudgetExhausted, LlmCall, LlmSpendDay, UserBudgetExhausted
from app.llm.embeddings import GatewayEmbeddings
from app.llm.gateway import Gateway
from app.settings import Settings

pytestmark = pytest.mark.anyio

MODEL = "openai:text-embedding-3-small"
DAY = date(2026, 1, 3)


class FakeEmbeddingsApi:
    """Answers like OpenAI's embeddings endpoint: one vector per input, usage in tokens."""

    def __init__(self, tokens_per_input: int = 7, error: Exception | None = None) -> None:
        self.requests: list[list[str]] = []
        self.tokens_per_input = tokens_per_input
        self.error = error

    async def create(self, *, model: str, input: list[str], **kwargs: Any) -> Any:
        self.requests.append(list(input))
        if self.error:
            raise self.error
        data = [SimpleNamespace(index=i, embedding=[float(i)] * 3) for i in range(len(input))]
        usage = SimpleNamespace(prompt_tokens=self.tokens_per_input * len(input))
        return SimpleNamespace(data=list(reversed(data)), usage=usage)


@pytest.fixture
def embeddings(
    db_session: AsyncSession,
) -> Callable[..., tuple[GatewayEmbeddings, FakeEmbeddingsApi]]:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def make(
        cap: int = 1_000_000, user_cap: int | None = None, **api: Any
    ) -> tuple[GatewayEmbeddings, FakeEmbeddingsApi]:
        fake = FakeEmbeddingsApi(**api)
        return (
            GatewayEmbeddings(
                model=MODEL,
                price=prices.EMBEDDING_PRICES[MODEL],
                client=SimpleNamespace(embeddings=fake),
                cap_micro_usd=cap,
                user_cap_micro_usd=user_cap,
                sessionmaker=lambda: sessions,
                today=lambda: DAY,
            ),
            fake,
        )

    return make


async def day_calls(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def test_documents_are_embedded_in_order_and_settled_at_their_real_cost(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    gateway_embeddings, _ = embeddings()
    vectors = await gateway_embeddings.aembed_documents(["a", "b", "c"], feature="ingest")
    assert vectors == [[0.0] * 3, [1.0] * 3, [2.0] * 3]
    [call] = await day_calls(db_session)
    assert (call.status, call.model, call.feature) == ("settled", MODEL, "ingest")
    assert (call.input_tokens, call.output_tokens) == (21, 0)
    # 21 tokens at $0.02 per million, rounded up to a whole micro-USD.
    assert call.cost_micro_usd == 1
    assert (
        await db_session.scalar(
            select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
        )
        == 1
    )


async def test_a_query_is_one_metered_call(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    gateway_embeddings, fake = embeddings()
    assert await gateway_embeddings.aembed_query("what is rag?") == [0.0] * 3
    assert fake.requests == [["what is rag?"]]
    assert len(await day_calls(db_session)) == 1


async def test_an_embedding_the_budget_cant_cover_is_never_sent(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    gateway_embeddings, fake = embeddings(cap=0)
    with pytest.raises(BudgetExhausted):
        await gateway_embeddings.aembed_documents(["a"])
    assert fake.requests == []
    assert await day_calls(db_session) == []


async def test_one_users_cap_applies_to_embeddings(embeddings: Callable[..., Any]) -> None:
    gateway_embeddings, fake = embeddings(user_cap=0)
    with pytest.raises(UserBudgetExhausted):
        await gateway_embeddings.aembed_documents(["a"], user_ref="u_1")
    assert fake.requests == []


async def test_a_request_the_provider_rejected_releases_its_reservation(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    rejected = openai.BadRequestError(
        "too long",
        response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")),
        body=None,
    )
    gateway_embeddings, _ = embeddings(error=rejected)
    # The same family of errors as chat calls, so app/llm/errors.py answers it the same way.
    with pytest.raises(ModelInvalidRequestError):
        await gateway_embeddings.aembed_documents(["a"])
    [call] = await day_calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("failed", 0)


async def test_a_request_that_may_have_been_processed_keeps_its_reservation(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "http://x"))
    gateway_embeddings, _ = embeddings(error=timeout)
    with pytest.raises(ModelTimeoutError):
        await gateway_embeddings.aembed_documents(["a"])
    [call] = await day_calls(db_session)
    assert call.status == "failed"
    assert call.cost_micro_usd == call.reserved_micro_usd > 0


async def test_large_inputs_are_split_into_batches_each_metered(
    db_session: AsyncSession, embeddings: Callable[..., Any]
) -> None:
    gateway_embeddings, fake = embeddings()
    texts = [f"text {i}" for i in range(GatewayEmbeddings.MAX_INPUTS + 1)]
    vectors = await gateway_embeddings.aembed_documents(texts)
    assert [len(r) for r in fake.requests] == [GatewayEmbeddings.MAX_INPUTS, 1]
    assert len(vectors) == len(texts)
    assert len(await day_calls(db_session)) == 2


async def test_the_synchronous_methods_refuse_because_the_budget_cant_stop_them(
    embeddings: Callable[..., Any],
) -> None:
    gateway_embeddings, fake = embeddings()
    with pytest.raises(RuntimeError, match="aembed_documents"):
        gateway_embeddings.embed_documents(["a"])
    with pytest.raises(RuntimeError, match="aembed_query"):
        gateway_embeddings.embed_query("a")
    assert fake.requests == []


def test_an_embedding_model_without_a_price_is_refused_at_startup() -> None:
    settings = Settings(
        _env_file=None, app_env="test", embedding_model="openai:text-embedding-ada-002"
    )
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]
