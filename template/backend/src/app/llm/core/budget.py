"""The daily model budget: reserve before each call, settle after it, one llm_call row per call.

Before a call goes out, its Meter reserves the most it can cost (prices.ceiling) against today's
total in one atomic statement, so concurrent calls in any number of workers can't overshoot
LLM_DAILY_BUDGET_USD: a reservation that would cross it is refused with a 503 and the call is
never sent. When the call ends, its row is settled with the real cost from its Usage and the day's
total moves by the difference.

A call the provider rejected (core/errors.py REJECTED: 401, 403, 400, 404, 429, context overflow)
is settled at zero: it was never processed. Any other failure (a timeout, a dropped connection, a
5xx, a stream that broke midway) may still be billed, so it keeps its reservation, as does a call
whose usage is missing or whose settlement fails. Every doubt errs towards counting too much. Days
are UTC.

Framework-neutral: an adapter drives a Meter from its framework's call events
(app/llm/langchain/callbacks.py BudgetGuard for LangChain) and hands it neutral values.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Integer,
    String,
    false,
    func,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.llm.core import metrics, prices
from app.llm.core.bounds import check_prompt_size
from app.llm.core.errors import REJECTED, FailureKind
from app.llm.core.usage import Usage
from app.problems import Problem

__all__ = [
    "WARN_SHARE",
    "BudgetExhausted",
    "CallTags",
    "LlmCall",
    "LlmSpendDay",
    "LlmSpendUserDay",
    "Meter",
    "UserBudgetExhausted",
    "expire_stale",
    "reserve",
    "reserve_for",
    "settle",
    "utc_today",
]

log = structlog.get_logger()

# Spend that triggers the day's one warning log, as a share of the cap.
WARN_SHARE = 0.8


class LlmSpendDay(Base):
    """Reserved plus settled spend for one UTC day."""

    __tablename__ = "llm_spend_day"

    day: Mapped[date] = mapped_column(primary_key=True)
    committed_micro_usd: Mapped[int] = mapped_column(BigInteger, server_default="0")
    # Set by the first reservation to reach WARN_SHARE of the cap, so the day warns once even
    # though settlements below their reservation lower the total again.
    warned: Mapped[bool] = mapped_column(server_default=false())


class LlmSpendUserDay(Base):
    """Reserved plus settled spend for one user on one UTC day, when a per-user cap is set."""

    __tablename__ = "llm_spend_user_day"

    day: Mapped[date] = mapped_column(primary_key=True)
    user_ref: Mapped[str] = mapped_column(String(200), primary_key=True)
    committed_micro_usd: Mapped[int] = mapped_column(BigInteger, server_default="0")


class LlmCall(Base):
    """One model call: its reservation, then what it really used and cost."""

    __tablename__ = "llm_call"
    __table_args__ = (
        CheckConstraint("status IN ('reserved', 'settled', 'failed')", name="status"),
    )

    # The adapter's call id: for LangChain its run id, which also names the call's LangSmith trace.
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
    # Parts of the totals above that are priced or behave differently.
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    # From the call's metadata: what the product used the model for, and an opaque id of whom
    # for (never a name or an email), so cost can be read per feature and per user.
    feature: Mapped[str | None] = mapped_column(String(100), index=True)
    user_ref: Mapped[str | None] = mapped_column(String(200))
    # Whether this call reserved against the user's daily cap, so settlement adjusts that total
    # only for calls that added to it (a worker without the cap may serve the same user).
    user_reserved: Mapped[bool] = mapped_column(server_default=false())


class BudgetExhausted(Problem):
    def __init__(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        super().__init__(
            503,
            detail="The daily model budget is used up. Try again tomorrow (UTC).",
            headers={"Retry-After": str(max(1, int((midnight - now).total_seconds())))},
        )


class UserBudgetExhausted(Problem):
    """One user's daily share is used up; everyone else can still call the model."""

    def __init__(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        super().__init__(
            429,
            detail="Your daily model budget is used up. Try again tomorrow (UTC).",
            headers={"Retry-After": str(max(1, int((midnight - now).total_seconds())))},
        )


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
    feature: str | None = None,
    user_ref: str | None = None,
    user_reserved: bool = False,
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
    committed = (await session.execute(upsert)).scalar()
    if committed is None:
        return False
    # Once per day: the first reservation to bring the total to WARN_SHARE of the cap claims the
    # day's warning (one UPDATE, so only one process can win it).
    if committed >= cap * WARN_SHARE and await _claim_warning(session, day):
        log.warning(
            "llm_budget_nearly_used",
            day=day.isoformat(),
            committed_micro_usd=committed,
            cap_micro_usd=cap,
        )
    session.add(
        LlmCall(
            id=call_id,
            day=day,
            model=model,
            prompt=prompt,
            prompt_sha=prompt_sha,
            feature=feature,
            user_ref=user_ref,
            user_reserved=user_reserved,
            status="reserved",
            reserved_micro_usd=amount,
        )
    )
    return True


async def _claim_warning(session: AsyncSession, day: date) -> bool:
    claimed = await session.execute(
        update(LlmSpendDay)
        .where(LlmSpendDay.day == day, LlmSpendDay.warned.is_(False))
        .values(warned=True)
        .returning(LlmSpendDay.day)
    )
    return claimed.first() is not None


async def settle(
    session: AsyncSession,
    *,
    call_id: uuid.UUID,
    status: str,
    cost: int | None,
    usage: Usage | None = None,
    latency_ms: int | None = None,
) -> None:
    """Record what a reserved call really cost (None keeps the reservation as its cost) and move
    the day's total by the difference. Settling twice changes nothing. Without usage (None) the
    token columns stay unknown (NULL)."""
    row = (
        await session.execute(
            update(LlmCall)
            .where(LlmCall.id == call_id, LlmCall.status == "reserved")
            .values(
                status=status,
                settled_at=func.now(),
                cost_micro_usd=LlmCall.reserved_micro_usd if cost is None else cost,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                latency_ms=latency_ms,
                **prices.token_details(usage),
            )
            .returning(
                LlmCall.day,
                LlmCall.user_ref,
                LlmCall.user_reserved,
                LlmCall.reserved_micro_usd,
                LlmCall.cost_micro_usd,
            )
        )
    ).first()
    if row is None:
        return
    day, user_ref, user_reserved, reserved, charged = row
    difference = (charged or 0) - reserved
    # The user's row before the day's: the order reservations lock them in, so a settlement and
    # a reservation for the same user can't deadlock.
    if user_reserved:
        await session.execute(
            update(LlmSpendUserDay)
            .where(LlmSpendUserDay.day == day, LlmSpendUserDay.user_ref == user_ref)
            .values(committed_micro_usd=LlmSpendUserDay.committed_micro_usd + difference)
        )
    await session.execute(
        update(LlmSpendDay)
        .where(LlmSpendDay.day == day)
        .values(committed_micro_usd=LlmSpendDay.committed_micro_usd + difference)
    )


class _Refused(Exception):
    def __init__(self, scope: str) -> None:
        self.scope = scope


async def reserve_for(
    session: AsyncSession,
    *,
    user_ref: str | None,
    user_cap: int | None,
    **call: Any,
) -> str | None:
    """reserve(), plus the user's own daily cap when one is set. Both totals move together or
    neither does (one savepoint). Rows are locked user first, then day; settle() keeps that
    order. Returns None when reserved, else what refused it: "day" or
    "user"."""
    try:
        async with session.begin_nested():
            capped = user_ref is not None and user_cap is not None
            if capped and not await _add_user_spend(
                session, call["day"], str(user_ref), call["amount"], int(user_cap or 0)
            ):
                raise _Refused("user")
            if not await reserve(session, user_ref=user_ref, user_reserved=capped, **call):
                raise _Refused("day")
    except _Refused as refused:
        return refused.scope
    return None


async def _add_user_spend(
    session: AsyncSession, day: date, user_ref: str, amount: int, cap: int
) -> bool:
    if amount > cap:
        return False
    new = insert(LlmSpendUserDay).values(day=day, user_ref=user_ref, committed_micro_usd=amount)
    total = LlmSpendUserDay.committed_micro_usd + new.excluded.committed_micro_usd
    upsert = new.on_conflict_do_update(
        index_elements=[LlmSpendUserDay.day, LlmSpendUserDay.user_ref],
        set_={"committed_micro_usd": total},
        where=total <= cap,
    ).returning(LlmSpendUserDay.committed_micro_usd)
    return (await session.execute(upsert)).first() is not None


async def expire_stale(session: AsyncSession, *, older_than: timedelta) -> int:
    """Close reservations left open by a process that died mid-call (a crash, a SIGKILL after
    the grace period). Whether the provider billed them is unknown, so each keeps its
    reservation as its cost (the day's total already counts it) and is marked failed. Returns how
    many were closed. Run it from a scheduled job, or
    `python -m app.llm.core.report --expire-stale`."""
    closed = await session.execute(
        update(LlmCall)
        .where(LlmCall.status == "reserved", LlmCall.created_at < func.now() - older_than)
        .values(status="failed", settled_at=func.now(), cost_micro_usd=LlmCall.reserved_micro_usd)
        .returning(LlmCall.id)
    )
    count = len(closed.all())
    if count:
        log.warning("llm_reservations_expired", count=count)
    return count


@dataclass(frozen=True, slots=True, kw_only=True)
class CallTags:
    """What a call is for, recorded on its llm_call row: the prompt's name and version, the
    product feature, and an opaque id of whom for (never a name or an email; counted against that
    user's daily cap when one is set). Only prompt and feature become metric labels."""

    prompt: str | None = None
    prompt_sha: str | None = None
    feature: str | None = None
    user_ref: str | None = None


class Meter:
    """The budget for one model, driven by an adapter: reserve() before a call is sent, then
    settle() when it ends, failed() when it fails or abandon() if it is cancelled. Each call gets
    one llm_call row and adds to the llm.* metrics; the call id is the adapter's (unique per call:
    it is the row's key).

    The settings are plain attributes, read on every call (tests and the gateway may change them).
    `sessionmaker` is looked up per call, so tests can point the app at their own transaction;
    `today` and `clock` (seconds, for latency) are there to be replaced in tests.
    """

    def __init__(
        self,
        *,
        model: str,
        price: prices.Price,
        cap_micro_usd: int,
        user_cap_micro_usd: int | None = None,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        today: Callable[[], date] = utc_today,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.model = model
        self.price = price
        self.cap = cap_micro_usd
        # Each user's share of a day, enforced for calls tagged with a user_ref.
        self.user_cap = user_cap_micro_usd
        self.sessionmaker = sessionmaker
        self.today = today
        # Looked up on every call (not bound here), so a test may patch time.monotonic.
        self.clock = clock if clock is not None else (lambda: time.monotonic())
        # Per call in flight: when it started, and what the metrics break it down by.
        self._started: dict[uuid.UUID, tuple[float, dict[str, str]]] = {}

    async def reserve(
        self,
        call_id: uuid.UUID,
        *,
        input_tokens: int,
        output_tokens: int,
        tags: CallTags | None = None,
    ) -> int:
        """Reserve the most the call can cost: `input_tokens` and `output_tokens` are upper bounds
        (core/bounds.py). Returns the amount reserved, in micro-USD. Raises PromptTooLong for a
        prompt longer than its price covers, UserBudgetExhausted (429) when the user's cap would be
        crossed, BudgetExhausted (503) when the day's would; nothing is reserved then."""
        check_prompt_size(self.price, input_tokens)
        amount = prices.ceiling(self.price, input_tokens, output_tokens)
        tags = tags or CallTags()
        async with self.sessionmaker()() as session, session.begin():
            refused = await reserve_for(
                session,
                user_cap=self.user_cap,
                call_id=call_id,
                day=self.today(),
                model=self.model,
                amount=amount,
                cap=self.cap,
                prompt=tags.prompt,
                prompt_sha=tags.prompt_sha,
                feature=tags.feature,
                user_ref=tags.user_ref,
            )
        labels = metrics.labels(self.model, tags.prompt, tags.feature)
        if refused is None:
            self._started[call_id] = (self.clock(), labels)
            return amount
        metrics.record_refusal(labels)
        log.warning(
            "llm_budget_exhausted", scope=refused, model=self.model, reservation_micro_usd=amount
        )
        raise UserBudgetExhausted if refused == "user" else BudgetExhausted

    async def settle(self, call_id: uuid.UUID, usage: Usage | None) -> None:
        """The call ended: settle it at its real cost. Without usage (None) it keeps its
        reservation as its cost and a warning is logged."""
        if usage is None:
            log.warning("llm_usage_missing", model=self.model, call_id=str(call_id))
        cost = None if usage is None else prices.cost(self.price, usage)
        await self._close(call_id, "settled", cost, usage)

    async def failed(self, call_id: uuid.UUID, kind: FailureKind | None, *, streamed: bool) -> None:
        """The call failed. A kind the provider refuses before doing any work (REJECTED) settles
        at zero, unless output had already streamed; anything else, an unknown kind (None)
        included, keeps the reservation as its cost."""
        rejected = kind in REJECTED and not streamed
        await self._close(call_id, "failed", 0 if rejected else None, None)

    def abandon(self, call_id: uuid.UUID) -> None:
        """Stop tracking a call that ended without settle() or failed(), one cancelled mid-flight.
        Whether it was billed is unknown, so its row stays reserved (counted at its reservation)
        until expire_stale closes it. Never awaits, so it is safe while the task is being
        cancelled."""
        self._started.pop(call_id, None)

    async def _close(
        self, call_id: uuid.UUID, status: str, cost: int | None, usage: Usage | None
    ) -> None:
        started, labels = self._started.pop(call_id, (None, metrics.labels(self.model)))
        latency_ms = round((self.clock() - started) * 1000) if started is not None else None
        # The call has happened by now: a settlement that fails must not fail it, and the
        # reservation it leaves in place overstates the spend rather than hiding it.
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
