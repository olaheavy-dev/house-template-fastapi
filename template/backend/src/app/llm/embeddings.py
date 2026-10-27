"""Embeddings under the daily model budget, like every chat call.

LangChain's embedding classes discard the provider's usage and take no callbacks, so the gateway
calls OpenAI's embeddings endpoint itself: each request is reserved before it is sent (its UTF-8
bytes bound its tokens, as in budget.input_bound) and settled from the usage OpenAI returns. A
request the provider rejected releases its reservation; any other failure keeps it. Errors come
back as LangChain's Model*Error family, so app/llm/errors.py answers them as it does chat calls.

Ask the gateway for it and always await it:

    vectors = await gateway.embeddings().aembed_documents(texts, feature="ingest")
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from typing import Any

import openai
import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages.ai import UsageMetadata
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import metrics, prices
from app.llm.budget import (
    INPUT_MARGIN_TOKENS,
    BudgetExhausted,
    UserBudgetExhausted,
    reserve_for,
    settle,
    utc_today,
)

log = structlog.get_logger()

# Statuses the provider answers before doing any work: nothing was billed.
_REJECTED = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.RateLimitError,
    openai.BadRequestError,
)
# Most specific first: APITimeoutError is an APIConnectionError.
_AS_MODEL_ERROR: tuple[tuple[type[Exception], type[ModelError]], ...] = (
    (openai.AuthenticationError, ModelAuthenticationError),
    (openai.PermissionDeniedError, ModelPermissionDeniedError),
    (openai.NotFoundError, ModelNotFoundError),
    (openai.RateLimitError, ModelRateLimitError),
    (openai.BadRequestError, ModelInvalidRequestError),
    (openai.APITimeoutError, ModelTimeoutError),
    (openai.APIConnectionError, ModelConnectionError),
    (openai.APIError, ModelAPIError),
)


def openai_client(api_key: SecretStr, *, timeout: float, max_retries: int) -> openai.AsyncOpenAI:
    """OpenAI's client with the gateway's timeout and its SDK retries as the only retry layer."""
    return openai.AsyncOpenAI(
        api_key=api_key.get_secret_value(), timeout=timeout, max_retries=max_retries
    )


def _as_model_error(error: Exception) -> Exception:
    for source, target in _AS_MODEL_ERROR:
        if isinstance(error, source):
            # The type only: a provider's message can echo the input.
            return target(f"OpenAI embeddings: {type(error).__name__}")
    return error


class GatewayEmbeddings(Embeddings):
    """OpenAI embeddings, each request reserved and settled against the daily budget."""

    # OpenAI's limits per request: 2,048 inputs and 300,000 tokens (bytes bound tokens).
    MAX_INPUTS = 2048
    MAX_BYTES = 300_000 - INPUT_MARGIN_TOKENS

    def __init__(
        self,
        *,
        model: str,
        price: prices.Price,
        client: Any,
        cap_micro_usd: int,
        user_cap_micro_usd: int | None = None,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        today: Callable[[], date] = utc_today,
    ) -> None:
        self.model = model
        self._model_id = model.split(":", 1)[1]
        self.price = price
        self._client = client
        self.cap = cap_micro_usd
        self.user_cap = user_cap_micro_usd
        self.sessionmaker = sessionmaker
        self.today = today

    async def aembed_documents(
        self, texts: list[str], *, feature: str | None = None, user_ref: str | None = None
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors += await self._embed(batch, feature=feature, user_ref=user_ref)
        return vectors

    async def aembed_query(
        self, text: str, *, feature: str | None = None, user_ref: str | None = None
    ) -> list[float]:
        return (await self._embed([text], feature=feature, user_ref=user_ref))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("The budget can't stop a synchronous call: await aembed_documents().")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("The budget can't stop a synchronous call: await aembed_query().")

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        size = 0
        for text in texts:
            length = len(text.encode())
            if batch and (len(batch) == self.MAX_INPUTS or size + length > self.MAX_BYTES):
                yield batch
                batch, size = [], 0
            batch.append(text)
            size += length
        if batch:
            yield batch

    async def _embed(
        self, batch: list[str], *, feature: str | None, user_ref: str | None
    ) -> list[list[float]]:
        call_id = uuid.uuid4()
        bound = sum(len(text.encode()) for text in batch) + INPUT_MARGIN_TOKENS
        amount = prices.ceiling(self.price, bound, 0)
        labels = metrics.labels(self.model, None, feature)
        async with self.sessionmaker()() as session, session.begin():
            refused = await reserve_for(
                session,
                user_ref=user_ref,
                user_cap=self.user_cap,
                call_id=call_id,
                day=self.today(),
                model=self.model,
                amount=amount,
                cap=self.cap,
                feature=feature,
            )
        if refused is not None:
            metrics.record_refusal(labels)
            log.warning(
                "llm_budget_exhausted",
                scope=refused,
                model=self.model,
                reservation_micro_usd=amount,
            )
            raise UserBudgetExhausted if refused == "user" else BudgetExhausted
        started = time.monotonic()
        try:
            response = await self._client.embeddings.create(model=self._model_id, input=batch)
        except Exception as error:
            rejected = isinstance(error, _REJECTED)
            await self._settle(call_id, "failed", 0 if rejected else None, None, labels, started)
            raise _as_model_error(error) from error
        tokens = int(response.usage.prompt_tokens)
        usage: UsageMetadata = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}
        await self._settle(
            call_id, "settled", prices.cost(self.price, usage), usage, labels, started
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    async def _settle(
        self,
        call_id: uuid.UUID,
        status: str,
        cost: int | None,
        usage: UsageMetadata | None,
        labels: dict[str, str],
        started: float,
    ) -> None:
        latency_ms = round((time.monotonic() - started) * 1000)
        # As in BudgetGuard: a settlement that fails leaves the reservation, never hides spend.
        try:
            async with self.sessionmaker()() as session, session.begin():
                await settle(
                    session,
                    call_id=call_id,
                    status=status,
                    cost=cost,
                    usage=usage,
                    latency_ms=latency_ms,
                )
        except Exception:
            log.exception("llm_settle_failed", model=self.model, call_id=str(call_id))
        metrics.record_call(labels, status, cost, usage, latency_ms)
