"""Embeddings go through the same daily budget as chat calls: reserved before, settled from the
provider's usage after. The provider client is faked; the database is real."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import prices
from app.llm.core.budget import LlmCall
from app.llm.langchain.embeddings import GatewayEmbeddings
from app.llm.langchain.gateway import Gateway
from app.settings import Settings
from tests.models import EMBEDDING_MODEL

pytestmark = pytest.mark.anyio

MODEL = EMBEDDING_MODEL
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
                redactor=None,
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
    with pytest.raises(ValueError, match=r"EMBEDDING_MODEL.*EMBEDDING_PRICES"):
        Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]
