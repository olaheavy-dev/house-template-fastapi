"""The daily model budget: reserve before each call, settle after it, one row per call.

Before a call goes out, BudgetGuard reserves the most it can cost (prices.ceiling) against today's
total in one atomic statement, so concurrent calls in any number of workers can't overshoot
LLM_DAILY_BUDGET_USD: a reservation that would cross it is refused with a 503 and the call is
never sent. When the call ends, its row is settled with the real cost from usage_metadata and the
day's total moves by the difference.

A failed call is settled at zero (providers don't bill a request they rejected), unless it had
already streamed tokens; a call whose usage is missing, or whose settlement fails, keeps its
reservation. Both err towards counting too much. Days are UTC.

The guard is a LangChain callback attached to the model, so it covers every way of calling it
(ainvoke, astream, with_structured_output, tools, agents), but only when the call is awaited:
LangChain swallows errors from async callbacks on the synchronous path, so never call a gateway
model with invoke() or stream().
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Integer, String, func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.llm import prices
from app.problems import Problem

log = structlog.get_logger()

# Added to every reservation's input: message framing, and the system prompt a provider adds
# when tools are bound (Anthropic documents a few hundred tokens).
INPUT_MARGIN_TOKENS = 1024


class LlmSpendDay(Base):
    """Reserved plus settled spend for one UTC day."""

    __tablename__ = "llm_spend_day"

    day: Mapped[date] = mapped_column(primary_key=True)
    committed_micro_usd: Mapped[int] = mapped_column(BigInteger, server_default="0")


class LlmCall(Base):
    """One model call: its reservation, then what it really used and cost."""

    __tablename__ = "llm_call"
    __table_args__ = (
        CheckConstraint("status IN ('reserved', 'settled', 'failed')", name="status"),
    )

    # LangChain's run id, which also names the call's trace in LangSmith.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    day: Mapped[date] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(200))
    prompt: Mapped[str | None] = mapped_column(String(200))
    prompt_sha: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    reserved_micro_usd: Mapped[int] = mapped_column(BigInteger)
    cost_micro_usd: Mapped[int | None] = mapped_column(BigInteger)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)


class BudgetExhausted(Problem):
    def __init__(self) -> None:
        super().__init__(503, detail="The daily model budget is used up. Try again tomorrow (UTC).")


def utc_today() -> date:
    return datetime.now(UTC).date()


async def reserve(
    session: AsyncSession,
    *,
    call_id: uuid.UUID,
    day: date,
    model: str,
    amount: int,
    cap: int,
    prompt: str | None = None,
    prompt_sha: str | None = None,
) -> bool:
    """Add `amount` to the day's total and record the call, or change nothing and return False if
    that would cross `cap`. One statement: Postgres locks the day's row, so concurrent
    reservations queue rather than overshoot."""
    if amount > cap:
        return False
    new = insert(LlmSpendDay).values(day=day, committed_micro_usd=amount)
    total = LlmSpendDay.committed_micro_usd + new.excluded.committed_micro_usd
    upsert = new.on_conflict_do_update(
        index_elements=[LlmSpendDay.day],
        set_={"committed_micro_usd": total},
        where=total <= cap,
    ).returning(LlmSpendDay.committed_micro_usd)
    if (await session.execute(upsert)).first() is None:
        return False
    session.add(
        LlmCall(
            id=call_id,
            day=day,
            model=model,
            prompt=prompt,
            prompt_sha=prompt_sha,
            status="reserved",
            reserved_micro_usd=amount,
        )
    )
    return True


async def settle(
    session: AsyncSession,
    *,
    call_id: uuid.UUID,
    status: str,
    cost: int | None,
    usage: UsageMetadata | None = None,
) -> None:
    """Record what a reserved call really cost (None keeps the reservation as its cost) and move
    the day's total by the difference. Settling twice changes nothing."""
    row = (
        await session.execute(
            update(LlmCall)
            .where(LlmCall.id == call_id, LlmCall.status == "reserved")
            .values(
                status=status,
                settled_at=func.now(),
                cost_micro_usd=LlmCall.reserved_micro_usd if cost is None else cost,
                input_tokens=usage["input_tokens"] if usage else None,
                output_tokens=usage["output_tokens"] if usage else None,
            )
            .returning(LlmCall.day, LlmCall.reserved_micro_usd, LlmCall.cost_micro_usd)
        )
    ).first()
    if row is None:
        return
    day, reserved, charged = row
    await session.execute(
        update(LlmSpendDay)
        .where(LlmSpendDay.day == day)
        .values(committed_micro_usd=LlmSpendDay.committed_micro_usd + ((charged or 0) - reserved))
    )


def input_bound(messages: Sequence[BaseMessage], invocation_params: dict[str, Any]) -> int:
    """More input tokens than the request can hold: its UTF-8 bytes (byte-level BPE tokenizers
    never use more than one token per byte) plus a margin. Tools and schemas are in the params."""
    payload = json.dumps(
        [[m.model_dump() for m in messages], invocation_params], default=str, ensure_ascii=False
    )
    return len(payload.encode()) + INPUT_MARGIN_TOKENS


def _usage(response: LLMResult) -> UsageMetadata | None:
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration):
                usage = getattr(generation.message, "usage_metadata", None)
                if usage:
                    return usage  # type: ignore[no-any-return]
    return None


class BudgetGuard(AsyncCallbackHandler):
    """Reserves before each call of the model it's attached to, and settles after it."""

    raise_error = True  # a refused reservation must stop the call
    run_inline = True

    def __init__(
        self,
        *,
        model: str,
        price: prices.Price,
        cap_micro_usd: int,
        max_tokens: int,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        today: Callable[[], date] = utc_today,
    ) -> None:
        self.model = model
        self.price = price
        self.cap = cap_micro_usd
        self.max_tokens = max_tokens
        # Looked up per call, so tests can point the app at their own transaction.
        self.sessionmaker = sessionmaker
        self.today = today
        self._streamed: set[uuid.UUID] = set()

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
        # A caller may bind a larger limit than the gateway's; reserve for whichever is larger.
        bound = [params.get(k) for k in ("max_tokens", "max_completion_tokens")]
        output = max([self.max_tokens, *(b for b in bound if isinstance(b, int))])
        amount = prices.ceiling(
            self.price, sum(input_bound(batch, params) for batch in messages), output
        )
        metadata = metadata or {}
        async with self.sessionmaker()() as session, session.begin():
            reserved = await reserve(
                session,
                call_id=run_id,
                day=self.today(),
                model=self.model,
                amount=amount,
                cap=self.cap,
                prompt=metadata.get("prompt"),
                prompt_sha=metadata.get("prompt_sha"),
            )
        if not reserved:
            log.warning("llm_budget_exhausted", model=self.model, reservation_micro_usd=amount)
            raise BudgetExhausted

    async def on_llm_new_token(
        self, token: str | list[str | dict[str, Any]], *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._streamed.add(run_id)

    async def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._streamed.discard(run_id)
        usage = _usage(response)
        if usage is None:
            log.warning("llm_usage_missing", model=self.model, call_id=str(run_id))
        cost = prices.cost(self.price, usage) if usage else None
        await self._settle(run_id, "settled", cost, usage)

    async def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        streamed = run_id in self._streamed
        self._streamed.discard(run_id)
        await self._settle(run_id, "failed", None if streamed else 0, None)

    async def _settle(
        self, run_id: uuid.UUID, status: str, cost: int | None, usage: UsageMetadata | None
    ) -> None:
        # The call has happened by now: a settlement that fails must not fail it, and the
        # reservation it leaves in place overstates the spend rather than hiding it.
        try:
            async with self.sessionmaker()() as session, session.begin():
                await settle(session, call_id=run_id, status=status, cost=cost, usage=usage)
        except Exception:
            log.exception("llm_settle_failed", model=self.model, call_id=str(run_id))
