"""Embeddings under the daily model budget, like every chat call.

LangChain's embedding classes discard the provider's usage and take no callbacks, so the gateway
calls OpenAI's embeddings endpoint itself (core/openai_embeddings.py): each request is reserved
before it is sent (its UTF-8 bytes bound its tokens, as in core/bounds.py) and settled from the
usage OpenAI returns, by core/embeddings.py on a Meter. A request the provider rejected releases
its reservation; any other failure keeps it (both decided by the error's core/errors.py
FailureKind). An embedding price's max_prompt_tokens applies to each request: texts are batched
under it, and one text past it alone is refused with a 422. Errors come back as LangChain's
Model*Error family, so app/llm/langchain/errors.py answers them as it does chat calls.

Ask the gateway for it and always await it:

    vectors = await gateway.embeddings().aembed_documents(texts, feature="ingest")
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core import embeddings as core_embeddings
from app.llm.core import openai_embeddings
from app.llm.core.budget import CallTags, Meter, utc_today
from app.llm.core.errors import ProviderError
from app.llm.core.prices import Price
from app.llm.core.usage import Usage
from app.llm.langchain.errors import model_error


class GatewayEmbeddings(Embeddings):
    """OpenAI embeddings, each request reserved and settled against the daily budget through its
    Meter. model, price, cap, user_cap, sessionmaker and today are the Meter's."""

    # OpenAI's limits per request: 2,048 inputs and 300,000 tokens (bytes bound tokens).
    MAX_INPUTS = core_embeddings.MAX_INPUTS
    MAX_BYTES = core_embeddings.MAX_BYTES

    def __init__(
        self,
        *,
        model: str,
        price: Price,
        client: Any,
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

    @property
    def model(self) -> str:
        return self.meter.model

    @model.setter
    def model(self, value: str) -> None:
        self.meter.model = value

    @property
    def price(self) -> Price:
        return self.meter.price

    @price.setter
    def price(self, value: Price) -> None:
        self.meter.price = value

    @property
    def cap(self) -> int:
        return self.meter.cap

    @cap.setter
    def cap(self, value: int) -> None:
        self.meter.cap = value

    @property
    def user_cap(self) -> int | None:
        return self.meter.user_cap

    @user_cap.setter
    def user_cap(self, value: int | None) -> None:
        self.meter.user_cap = value

    @property
    def sessionmaker(self) -> Callable[[], async_sessionmaker[AsyncSession]]:
        return self.meter.sessionmaker

    @sessionmaker.setter
    def sessionmaker(self, value: Callable[[], async_sessionmaker[AsyncSession]]) -> None:
        self.meter.sessionmaker = value

    @property
    def today(self) -> Callable[[], date]:
        return self.meter.today

    @today.setter
    def today(self, value: Callable[[], date]) -> None:
        self.meter.today = value

    async def aembed_documents(
        self, texts: list[str], *, feature: str | None = None, user_ref: str | None = None
    ) -> list[list[float]]:
        return await self._embed(texts, CallTags(feature=feature, user_ref=user_ref))

    async def aembed_query(
        self, text: str, *, feature: str | None = None, user_ref: str | None = None
    ) -> list[float]:
        return (await self._embed([text], CallTags(feature=feature, user_ref=user_ref)))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("The budget can't stop a synchronous call: await aembed_documents().")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("The budget can't stop a synchronous call: await aembed_query().")

    async def _create(self, batch: list[str]) -> tuple[list[list[float]], Usage]:
        return await openai_embeddings.create(self._client, self._model_id, batch)

    async def _embed(self, texts: list[str], tags: CallTags) -> list[list[float]]:
        try:
            return await core_embeddings.embed(
                self.meter,
                self._create,
                texts,
                tags=tags,
                max_inputs=self.MAX_INPUTS,
                max_bytes=self.MAX_BYTES,
            )
        except ProviderError as error:
            # The type only: the core's error holds no provider text. Its cause, the provider's
            # own error, is this one's.
            raise model_error(error.kind, f"OpenAI embeddings: {error.source}") from (
                error.__cause__ or error
            )
