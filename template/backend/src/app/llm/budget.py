"""The daily model budget: reserve before each call, settle after it, one row per call.

Before a call goes out, BudgetGuard reserves the most it can cost (prices.ceiling) against today's
total in one atomic statement, so concurrent calls in any number of workers can't overshoot
LLM_DAILY_BUDGET_USD: a reservation that would cross it is refused with a 503 and the call is
never sent. When the call ends, its row is settled with the real cost from usage_metadata and the
day's total moves by the difference.

A call the provider rejected (401, 403, 400, 404, 429, context overflow) is settled at zero: it
was never processed. Any other failure (a timeout, a dropped connection, a 5xx, a stream that
broke midway) may still be billed, so it keeps its reservation, as does a call whose usage is
missing or whose settlement fails. Every doubt errs towards counting too much. Days are UTC.

Images are reserved at IMAGE_TOKENS each, however they are sent. Documents, audio and video have
no per-item bound (a PDF costs per page), so a call carrying one is refused before it is sent.

prices.py prices tokens at the standard tier only, so a call that would be billed for more is
refused before it is sent: one binding a tool the provider runs itself (web search, file search,
code execution: billed per use on top of tokens), or asking for a dearer service tier (priority).

The guard is a LangChain callback attached to the model, so it covers every way of calling it
(ainvoke, astream, with_structured_output, tools, agents), but only when the call is awaited:
LangChain swallows errors from async callbacks on the synchronous path, so never call a gateway
model with invoke() or stream().
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult
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
from app.llm import metrics, prices
from app.problems import Problem

log = structlog.get_logger()

# Added to every reservation's input: message framing, and the system prompt a provider adds
# when tools are bound (Anthropic documents a few hundred tokens).
INPUT_MARGIN_TOKENS = 1024
# Spend that triggers the day's one warning log, as a share of the cap.
WARN_SHARE = 0.8
# Per image, above Anthropic's documented maximum of 4,784 (high-resolution models, vision docs,
# checked 26 Oct 2026). Check the provider's image pricing before sending images to another one.
IMAGE_TOKENS = 6_000
_IMAGE_BLOCKS = {"image", "image_url", "input_image"}
_UNBOUNDED_BLOCKS = {"document", "file", "input_file", "audio", "input_audio", "video"}
# Errors that mean the provider rejected the request without processing it (LangChain's
# provider-independent classes, keyed to the HTTP status).
_REJECTED = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
    ContextOverflowError,
)


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
    usage: UsageMetadata | None = None,
    latency_ms: int | None = None,
) -> None:
    """Record what a reserved call really cost (None keeps the reservation as its cost) and move
    the day's total by the difference. Settling twice changes nothing."""
    details = prices.token_details(usage)
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
                latency_ms=latency_ms,
                **details,
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


class PromptTooLong(Problem):
    def __init__(self) -> None:
        super().__init__(422, detail="The input is too long for the language model.")


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
    many were closed. Run it from a scheduled job, or `python -m app.llm.report --expire-stale`."""
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


class UnboundedMedia(Problem):
    """A message carries content whose token cost the budget can't bound in advance."""

    def __init__(self, kind: str) -> None:
        super().__init__(422, detail="This kind of attachment can't be sent to the language model.")
        self.kind = kind


def _images(value: Any) -> int:
    """Image blocks anywhere in a message's content (tool results can hold them too)."""
    if isinstance(value, list):
        return sum(_images(item) for item in value)
    if not isinstance(value, dict):
        return 0
    kind = value.get("type")
    if kind in _IMAGE_BLOCKS:
        return 1
    if kind in _UNBOUNDED_BLOCKS:
        raise UnboundedMedia(str(kind))
    return sum(_images(item) for item in value.values())


def input_bound(messages: Sequence[BaseMessage], invocation_params: dict[str, Any]) -> int:
    """More input tokens than the request can hold: its UTF-8 bytes (byte-level BPE tokenizers
    never use more than one token per byte), IMAGE_TOKENS per image (an image sent by URL or file
    id is only a few bytes here), plus a margin. Tools and schemas are in the params."""
    payload = json.dumps(
        [[m.model_dump() for m in messages], invocation_params], default=str, ensure_ascii=False
    )
    images = sum(_images(m.content) for m in messages)
    return len(payload.encode()) + images * IMAGE_TOKENS + INPUT_MARGIN_TOKENS


# Tools the app runs itself, which cost only the tokens of the call: OpenAI's "function" and
# "custom" tools, and Anthropic's untyped ones. Any other type is a tool the provider runs.
_TOKEN_ONLY_TOOL_TYPES = frozenset({None, "function", "custom"})
# Service tiers billed at prices.py's standard rates or less (flex is cheaper, which only ever
# over-records).
_PRICED_SERVICE_TIERS = frozenset({None, "auto", "default", "flex", "standard_only"})


class UnpricedRequest(ValueError):
    """A call that would be billed for something prices.py doesn't price: a mistake in the code
    that made it, refused before anything is reserved or sent."""


def check_priced(invocation_params: dict[str, Any]) -> None:
    for tool in invocation_params.get("tools") or ():
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind not in _TOKEN_ONLY_TOOL_TYPES:
            raise UnpricedRequest(
                f"The provider runs the {kind!r} tool and bills it per use, which "
                "app/llm/prices.py doesn't price: use a tool the app runs instead."
            )
    tier = invocation_params.get("service_tier")
    if tier not in _PRICED_SERVICE_TIERS:
        raise UnpricedRequest(
            f"The {tier!r} service tier costs more than the standard rates app/llm/prices.py "
            "holds: add its prices before using it."
        )


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
        user_cap_micro_usd: int | None = None,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        today: Callable[[], date] = utc_today,
    ) -> None:
        self.model = model
        self.price = price
        self.cap = cap_micro_usd
        # Each user's share of a day, enforced for calls tagged with metadata user_ref.
        self.user_cap = user_cap_micro_usd
        self.max_tokens = max_tokens
        # Looked up per call, so tests can point the app at their own transaction.
        self.sessionmaker = sessionmaker
        self.today = today
        self._streamed: set[uuid.UUID] = set()
        # Per call in flight: when it started, and what the metrics break it down by.
        self._started: dict[uuid.UUID, tuple[float, dict[str, str]]] = {}

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
        check_priced(params)
        # A caller may bind a larger limit than the gateway's; reserve for whichever is larger.
        bound = [params.get(k) for k in ("max_tokens", "max_completion_tokens")]
        output = max([self.max_tokens, *(b for b in bound if isinstance(b, int))])
        given = sum(input_bound(batch, params) for batch in messages)
        if self.price.max_prompt_tokens and given > self.price.max_prompt_tokens:
            # The bound is bytes, an over-estimate of tokens: a prompt near the limit in tokens
            # may be refused, never one past it sent and under-recorded.
            raise PromptTooLong
        amount = prices.ceiling(self.price, given, output)
        metadata = metadata or {}
        async with self.sessionmaker()() as session, session.begin():
            refused = await reserve_for(
                session,
                user_cap=self.user_cap,
                call_id=run_id,
                day=self.today(),
                model=self.model,
                amount=amount,
                cap=self.cap,
                prompt=metadata.get("prompt"),
                prompt_sha=metadata.get("prompt_sha"),
                feature=metadata.get("feature"),
                user_ref=metadata.get("user_ref"),
            )
        labels = metrics.labels(self.model, metadata.get("prompt"), metadata.get("feature"))
        if refused is None:
            self._started[run_id] = (time.monotonic(), labels)
            return
        metrics.record_refusal(labels)
        log.warning(
            "llm_budget_exhausted", scope=refused, model=self.model, reservation_micro_usd=amount
        )
        raise UserBudgetExhausted if refused == "user" else BudgetExhausted

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
        rejected = isinstance(error, _REJECTED) and not streamed
        await self._settle(run_id, "failed", 0 if rejected else None, None)

    async def _settle(
        self, run_id: uuid.UUID, status: str, cost: int | None, usage: UsageMetadata | None
    ) -> None:
        started, labels = self._started.pop(run_id, (None, metrics.labels(self.model)))
        latency_ms = round((time.monotonic() - started) * 1000) if started is not None else None
        # The call has happened by now: a settlement that fails must not fail it, and the
        # reservation it leaves in place overstates the spend rather than hiding it.
        try:
            async with self.sessionmaker()() as session, session.begin():
                await settle(
                    session,
                    call_id=run_id,
                    status=status,
                    cost=cost,
                    usage=usage,
                    latency_ms=latency_ms,
                )
        except Exception:
            log.exception("llm_settle_failed", model=self.model, call_id=str(run_id))
        metrics.record_call(labels, status, cost, usage, latency_ms)
