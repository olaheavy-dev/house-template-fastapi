"""The daily budget as a LangChain callback: BudgetGuard reads what LangChain reports about a call
(its messages and parameters, streamed tokens, usage, errors) and drives the core's Meter
(app/llm/core/budget.py), which reserves, settles and records the call.

The guard is attached to the model, so it covers every way of calling it (ainvoke, astream,
with_structured_output, tools, agents), but only when the call is awaited: LangChain swallows
errors from async callbacks on the synchronous path, so never call a gateway model with invoke()
or stream().
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, GenerationChunk, LLMResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core import bounds, prices
from app.llm.core.budget import CallTags, Meter, utc_today
from app.llm.core.context import USAGE_KEY, usage_record
from app.llm.langchain.errors import kind_of
from app.llm.langchain.usage import from_langchain

__all__ = ["BudgetGuard", "input_bound"]


def input_bound(messages: Sequence[BaseMessage], invocation_params: dict[str, Any]) -> int:
    """More input tokens than one batch of LangChain messages can hold, with the call's parameters
    (tools and schemas are in them): core/bounds.py request_bound of their serialised form."""
    return bounds.request_bound(
        [[m.model_dump() for m in messages], invocation_params], [m.content for m in messages]
    )


def _usage(response: LLMResult) -> UsageMetadata | None:
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration):
                usage = getattr(generation.message, "usage_metadata", None)
                if usage:
                    return usage  # type: ignore[no-any-return]
    return None


def _record(response: LLMResult, checked: int) -> None:
    """Write the input bound the budget checked on the call's answer, where the context report
    reads it once the thread is stored (#46, app/llm/langchain/context.py). The answer's own
    `response_metadata` is stored with it by LangGraph and never sent back to the provider
    (tests/test_llm_langchain_context.py checks the request payloads). The usage stays in the
    answer's `usage_metadata`, where LangChain puts it."""
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration):
                metadata = generation.message.response_metadata
                metadata[USAGE_KEY] = usage_record(None, checked=checked)


class BudgetGuard(AsyncCallbackHandler):
    """Reserves before each call of the model it's attached to, and settles after it, through its
    Meter. model, price, cap, user_cap, sessionmaker and today are the Meter's."""

    raise_error = True  # a refused reservation must stop the call
    run_inline = True

    def __init__(
        self,
        *,
        model: str,
        price: prices.Price,
        cap_micro_usd: int,
        max_tokens: int,
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
        self.max_tokens = max_tokens
        self._streamed: set[uuid.UUID] = set()
        # The input bound checked for each call in flight, recorded on its answer when it ends.
        self._checked: dict[uuid.UUID, int] = {}

    @property
    def model(self) -> str:
        return self.meter.model

    @model.setter
    def model(self, value: str) -> None:
        self.meter.model = value

    @property
    def price(self) -> prices.Price:
        return self.meter.price

    @price.setter
    def price(self, value: prices.Price) -> None:
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

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params: dict[str, Any] = kwargs.get("invocation_params") or {}
        bounds.check_priced(params)
        # A caller may bind a larger limit than the gateway's; reserve for whichever is larger.
        output = bounds.output_bound(self.max_tokens, params)
        given = sum(input_bound(batch, params) for batch in messages)
        metadata = metadata or {}
        counted = metadata.get("redacted_count")
        await self.meter.reserve(
            run_id,
            input_tokens=given,
            output_tokens=output,
            tags=CallTags(
                prompt=metadata.get("prompt"),
                prompt_sha=metadata.get("prompt_sha"),
                feature=metadata.get("feature"),
                user_ref=metadata.get("user_ref"),
                # Set by the redacting wrapper (a count, never a value); absent when it is off.
                redacted_count=counted if type(counted) is int else None,
            ),
        )
        self._checked[run_id] = given

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._streamed.add(run_id)
        # A streamed answer is merged from its chunks by whoever reads the stream (LangGraph, the
        # redacting wrapper), and on_llm_end only sees a merged copy of its own. This chunk is the
        # object the caller gets next, so the record goes on the first one and is merged into the
        # stored answer.
        checked = self._checked.pop(run_id, None)
        if checked is not None and isinstance(chunk, ChatGenerationChunk):
            chunk.message.response_metadata[USAGE_KEY] = usage_record(None, checked=checked)

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._streamed.discard(run_id)
        # A streamed call's record is already on its first chunk (on_llm_new_token).
        checked = self._checked.pop(run_id, None)
        if checked is not None:
            _record(response, checked)
        usage = _usage(response)
        await self.meter.settle(run_id, from_langchain(usage) if usage else None)

    async def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        streamed = run_id in self._streamed
        self._streamed.discard(run_id)
        self._checked.pop(run_id, None)
        await self.meter.failed(run_id, kind_of(error), streamed=streamed)
