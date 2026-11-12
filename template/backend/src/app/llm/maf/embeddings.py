"""Embeddings under the daily model budget, like every chat call.

MAF has no embeddings client that reports the provider's usage to the budget, so the gateway calls
OpenAI's embeddings endpoint itself (core/openai_embeddings.py), exactly as the LangChain adapter
does: each request is reserved before it is sent (its UTF-8 bytes bound its tokens) and settled
from the usage OpenAI returns, by core/embeddings.py on a Meter. With redaction on (the default)
the texts are cleaned of personal data first, one way, each on its own (decision 0006): embed
documents and the queries that look for them under the same `redaction_scope`. A failure is the
core's ProviderError (no provider text), answered by app/llm/core/errors.py like a chat call's.

Ask the gateway for it and always await it:

    vectors = await gateway.embeddings().embed_documents(texts, feature="ingest")
    query = await gateway.embeddings().embed_query(question, user_ref=user.id)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core import embeddings as core_embeddings
from app.llm.core import openai_embeddings
from app.llm.core.budget import CallTags, Meter, utc_today
from app.llm.core.prices import Price
from app.llm.core.redaction import Redactor
from app.llm.core.usage import Usage

__all__ = ["GatewayEmbeddings"]


class GatewayEmbeddings:
    """OpenAI embeddings, each request reserved and settled against the daily budget through its
    Meter. `redactor` is the gateway's (None when LLM_REDACTION=off)."""

    def __init__(
        self,
        *,
        model: str,
        price: Price,
        client: Any,
        redactor: Redactor | None,
        cap_micro_usd: int,
        user_cap_micro_usd: int | None = None,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        today: Callable[[], date] = utc_today,
    ) -> None:
        self.meter = Meter(
            model=model,
            price=price,
            cap_micro_usd=cap_micro_usd,
            user_cap_micro_usd=user_cap_micro_usd,
            sessionmaker=sessionmaker,
            today=today,
        )
        self._model_id = model.split(":", 1)[1]
        self._client = client
        # None is redaction off. Required, so a forgotten argument is a TypeError, not a leak.
        self._redactor = redactor

    async def embed_documents(
        self, texts: list[str], *, feature: str | None = None, user_ref: str | None = None
    ) -> list[list[float]]:
        return await self._embed(texts, CallTags(feature=feature, user_ref=user_ref))

    async def embed_query(
        self, text: str, *, feature: str | None = None, user_ref: str | None = None
    ) -> list[float]:
        return (await self._embed([text], CallTags(feature=feature, user_ref=user_ref)))[0]

    async def aclose(self) -> None:
        """Closes the OpenAI client this class was given (the gateway calls it on shutdown)."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    async def _create(self, batch: list[str]) -> tuple[list[list[float]], Usage]:
        return await openai_embeddings.create(self._client, self._model_id, batch)

    async def _embed(self, texts: list[str], tags: CallTags) -> list[list[float]]:
        return await core_embeddings.embed(
            self.meter, self._create, texts, redactor=self._redactor, tags=tags
        )
