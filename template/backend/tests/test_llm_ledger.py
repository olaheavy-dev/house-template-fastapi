"""Characterisation of the budget ledger (#33): reserve, settle and failure paths, the daily cap,
the reservation amount, cost from usage details through the guard, and the spend report's grouping.
The existing test_llm_budget.py covers the main flows; these pin what it leaves open.

Money values are worked out by hand in the comments (Sonnet: input 2, output 10, cache read 0.1,
write 5m 2.5, write 1h 4 micro-USD per token; the highest input rate is 4, the 1h write).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from langchain_core.exceptions import (
    ContextOverflowError,
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm import prices, report
from app.llm.budget import (
    IMAGE_TOKENS,
    INPUT_MARGIN_TOKENS,
    BudgetExhausted,
    BudgetGuard,
    LlmCall,
    LlmSpendDay,
    LlmSpendUserDay,
    input_bound,
    reserve,
    reserve_for,
    settle,
)
from app.llm.embeddings import GatewayEmbeddings
from app.llm.gateway import Gateway
from app.settings import Settings
from tests.models import CHAT_MODEL, EMBEDDING_MODEL

pytestmark = pytest.mark.anyio

MODEL = "anthropic:claude-sonnet-5-5"
# Built here, so a project editing app/llm/prices.py breaks nothing: Sonnet-shaped rates in
# micro-USD per million tokens (input 2, output 10, cache read 0.1, write 5m 2.5, write 1h 4).
PRICE = prices.Price(
    prices.Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000),
    checked="2026-11-06",
    source="https://example.com/pricing",
)
DAY = date(2026, 1, 4)
HELLO: list[BaseMessage] = [HumanMessage("hello")]


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def guard(sessions: async_sessionmaker[AsyncSession]) -> Callable[..., BudgetGuard]:
    return lambda cap=10**9: BudgetGuard(
        model=MODEL,
        price=PRICE,
        cap_micro_usd=cap,
        max_tokens=500,
        sessionmaker=lambda: sessions,
        today=lambda: DAY,
    )


async def total(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


def end_of(usage: UsageMetadata | None) -> LLMResult:
    message = AIMessage("ok", usage_metadata=usage) if usage else AIMessage("ok")
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# --- reserve: the daily cap -----------------------------------------------------------------------


async def test_a_reservation_may_reach_the_cap_exactly_but_not_cross_it(
    db_session: AsyncSession,
) -> None:
    def take(amount: int) -> Any:
        return reserve(
            db_session, call_id=uuid.uuid4(), day=DAY, model=MODEL, amount=amount, cap=1_000
        )

    assert await take(600)
    assert await take(400)  # 600 + 400 = 1,000 = the cap: allowed (the rule is "<= cap")
    assert await total(db_session) == 1_000
    assert not await take(1)  # 1,001 > 1,000
    assert await total(db_session) == 1_000  # a refusal changes nothing
    assert len(await rows(db_session)) == 2  # and records no call


async def test_a_single_reservation_above_the_cap_is_refused_before_touching_the_day(
    db_session: AsyncSession,
) -> None:
    assert not await reserve(
        db_session, call_id=uuid.uuid4(), day=DAY, model=MODEL, amount=1_001, cap=1_000
    )
    assert await total(db_session) is None  # not even a zero row
    assert await rows(db_session) == []


async def test_a_new_reservation_is_recorded_as_reserved_with_its_tags(
    db_session: AsyncSession,
) -> None:
    call_id = uuid.uuid4()
    assert await reserve(
        db_session,
        call_id=call_id,
        day=DAY,
        model=MODEL,
        amount=250,
        cap=1_000,
        prompt="p",
        prompt_sha="abc",
        feature="f",
        user_ref="u",
    )
    [row] = await rows(db_session)
    assert (row.id, row.status, row.reserved_micro_usd, row.cost_micro_usd) == (
        call_id,
        "reserved",
        250,
        None,
    )
    assert (row.model, row.prompt, row.prompt_sha, row.feature, row.user_ref) == (
        MODEL,
        "p",
        "abc",
        "f",
        "u",
    )
    assert (row.user_reserved, row.input_tokens, row.settled_at) == (False, None, None)


async def test_the_warning_fires_at_exactly_80_percent_and_not_below(
    db_session: AsyncSession,
) -> None:
    def take(amount: int, day: date) -> Any:
        return reserve(
            db_session, call_id=uuid.uuid4(), day=day, model=MODEL, amount=amount, cap=1_000
        )

    with capture_logs() as logs:
        await take(799, date(2026, 1, 5))  # 79.9%
        assert not [log for log in logs if log["event"] == "llm_budget_nearly_used"]
        await take(800, date(2026, 1, 6))  # 80.0% (800 >= 1,000 x 0.8)
    [warning] = [log for log in logs if log["event"] == "llm_budget_nearly_used"]
    assert (warning["day"], warning["committed_micro_usd"], warning["cap_micro_usd"]) == (
        "2026-01-06",
        800,
        1_000,
    )


# --- the cap derived from settings ---------------------------------------------------------------


def gateway_with(**settings: Any) -> Gateway:
    given = Settings(
        _env_file=None, llm_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL, **settings
    )
    return Gateway(given, lambda: None)  # type: ignore[arg-type,return-value]


def test_the_daily_cap_is_the_settings_dollars_in_whole_micro_usd() -> None:
    # $5 x 1,000,000 = 5,000,000 micro-USD; the per-user cap is off unless set.
    guard = gateway_with(llm_daily_budget_usd=Decimal("5")).guard
    assert (guard.cap, guard.user_cap) == (5_000_000, None)
    # $0.25 x 1,000,000 = 250,000.
    both = gateway_with(
        llm_daily_budget_usd=Decimal("2"), llm_user_daily_budget_usd=Decimal("0.25")
    ).guard
    assert (both.cap, both.user_cap) == (2_000_000, 250_000)


def test_a_fraction_of_a_micro_usd_in_the_cap_is_cut_off_not_rounded() -> None:
    # int(usd x 1_000_000) truncates. The setting is a Decimal, so the product is exact:
    # 1.9999999 x 1,000,000 = 1,999,999.9 -> 1,999,999 (a rounding rule would give 2,000,000), and
    # 0.0000019 x 1,000,000 = 1.9 -> 1. This can only make the cap stricter.
    guard = gateway_with(
        llm_daily_budget_usd=Decimal("1.9999999"), llm_user_daily_budget_usd=Decimal("0.0000019")
    ).guard
    assert (guard.cap, guard.user_cap) == (1_999_999, 1)


def test_the_gateways_guard_reserves_for_the_models_max_tokens_and_price() -> None:
    guard = gateway_with(llm_max_tokens=321).guard
    assert (guard.model, guard.max_tokens) == (CHAT_MODEL, 321)
    assert guard.price is prices.PRICES[CHAT_MODEL]


# --- settle ---------------------------------------------------------------------------------------


async def test_settling_above_the_reservation_raises_the_days_total(
    db_session: AsyncSession,
) -> None:
    call_id = uuid.uuid4()
    assert await reserve(db_session, call_id=call_id, day=DAY, model=MODEL, amount=100, cap=1_000)
    await settle(db_session, call_id=call_id, status="settled", cost=130)
    # total = 100 reserved + (130 - 100) difference = 130
    assert await total(db_session) == 130
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd, row.settled_at is not None) == ("settled", 130, True)


async def test_settling_without_a_cost_keeps_the_reservation_and_without_usage_records_none(
    db_session: AsyncSession,
) -> None:
    call_id = uuid.uuid4()
    assert await reserve(db_session, call_id=call_id, day=DAY, model=MODEL, amount=400, cap=1_000)
    await settle(db_session, call_id=call_id, status="failed", cost=None, latency_ms=12)
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd, row.latency_ms) == ("failed", 400, 12)
    # No usage: token counts and details are unknown (None), not zero.
    assert (row.input_tokens, row.output_tokens) == (None, None)
    assert (row.reasoning_tokens, row.cache_read_tokens, row.cache_write_tokens) == (
        None,
        None,
        None,
    )
    assert await total(db_session) == 400


async def test_settling_at_zero_gives_the_whole_reservation_back(db_session: AsyncSession) -> None:
    call_id = uuid.uuid4()
    assert await reserve(db_session, call_id=call_id, day=DAY, model=MODEL, amount=400, cap=1_000)
    await settle(db_session, call_id=call_id, status="failed", cost=0)
    assert await total(db_session) == 0  # 400 + (0 - 400)


async def test_settling_an_unknown_call_changes_nothing(db_session: AsyncSession) -> None:
    await settle(db_session, call_id=uuid.uuid4(), status="settled", cost=5)
    assert await total(db_session) is None
    assert await rows(db_session) == []


async def test_a_users_total_moves_by_the_same_difference_as_the_days(
    db_session: AsyncSession,
) -> None:
    call_id = uuid.uuid4()
    refused = await reserve_for(
        db_session,
        user_ref="u_pin",
        user_cap=1_000,
        call_id=call_id,
        day=DAY,
        model=MODEL,
        amount=600,
        cap=10_000,
    )
    assert refused is None
    await settle(db_session, call_id=call_id, status="settled", cost=250)
    # 600 reserved, settled at 250: both totals move by 250 - 600 = -350.
    user = await db_session.scalar(
        select(LlmSpendUserDay.committed_micro_usd).where(LlmSpendUserDay.user_ref == "u_pin")
    )
    assert (user, await total(db_session)) == (250, 250)


async def test_a_reservation_over_the_users_cap_is_refused_as_user_and_equal_is_allowed(
    db_session: AsyncSession,
) -> None:
    def take(amount: int) -> Any:
        return reserve_for(
            db_session,
            user_ref="u_edge",
            user_cap=500,
            call_id=uuid.uuid4(),
            day=DAY,
            model=MODEL,
            amount=amount,
            cap=10_000,
        )

    assert await take(500) is None  # equal to the cap
    assert await take(1) == "user"  # 501 > 500
    assert await total(db_session) == 500  # the refused one never reached the day's total


# --- the guard: what it reserves and how failures settle -----------------------------------------


async def test_the_reservation_is_the_ceiling_of_the_input_bound_and_the_output_limit(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    g = guard()
    bound = input_bound(HELLO, {})
    # Sonnet ceiling: input at max(2, 4) = 4 micro-USD per token, output 10; both integers, so
    # bound x 4 + 500 x 10 = 4 x bound + 5,000 with no rounding.
    await g.on_chat_model_start({}, [HELLO], run_id=uuid.uuid4())
    # A larger limit bound by the caller is reserved for: 700 x 10 = 7,000.
    await g.on_chat_model_start(
        {}, [HELLO], run_id=uuid.uuid4(), invocation_params={"max_completion_tokens": 700}
    )
    # A smaller one, or a value that is not an integer, is ignored: the gateway's 500 stays.
    await g.on_chat_model_start(
        {}, [HELLO], run_id=uuid.uuid4(), invocation_params={"max_tokens": 100}
    )
    await g.on_chat_model_start(
        {}, [HELLO], run_id=uuid.uuid4(), invocation_params={"max_tokens": "9000"}
    )
    amounts = sorted(r.reserved_micro_usd for r in await rows(db_session))
    # The parameters are part of the input bound, so each call's input differs a little.
    assert 4 * bound + 5_000 in amounts
    assert input_bound(HELLO, {"max_completion_tokens": 700}) > bound
    assert 4 * input_bound(HELLO, {"max_completion_tokens": 700}) + 7_000 in amounts
    assert 4 * input_bound(HELLO, {"max_tokens": 100}) + 5_000 in amounts
    assert 4 * input_bound(HELLO, {"max_tokens": "9000"}) + 5_000 in amounts
    assert await total(db_session) == sum(amounts)


async def test_several_message_batches_are_reserved_as_one_sum(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    g = guard()
    await g.on_chat_model_start({}, [HELLO, HELLO], run_id=uuid.uuid4())
    [row] = await rows(db_session)
    assert row.reserved_micro_usd == 4 * (2 * input_bound(HELLO, {})) + 5_000


def test_the_input_bound_is_the_utf8_bytes_plus_images_plus_the_margin() -> None:
    # No messages and no params serialise to the 8 characters "[[], {}]" (8 bytes), plus the
    # 1,024 margin = 1,032.
    assert input_bound([], {}) == len("[[], {}]") + 1_024 == 1_032
    plain = input_bound([HumanMessage("e" * 100)], {})
    # "é" is two UTF-8 bytes, so 100 of them are exactly 100 bytes more than 100 "e": the bound
    # counts bytes, not characters.
    assert input_bound([HumanMessage("é" * 100)], {}) - plain == 100
    assert (INPUT_MARGIN_TOKENS, IMAGE_TOKENS) == (1_024, 6_000)
    # An image block and a same-length block of another type differ by IMAGE_TOKENS plus the
    # difference in the bytes of the type name: len("image") - len("other") = 0.
    image = HumanMessage(content=[{"type": "image", "url": "u"}])
    other = HumanMessage(content=[{"type": "other", "url": "u"}])
    assert input_bound([image], {}) - input_bound([other], {}) == IMAGE_TOKENS
    # A nested image (a tool result holding one) counts too.
    nested = HumanMessage(content=[{"type": "tool_result", "content": [{"type": "image_url"}]}])
    plain_nested = HumanMessage(content=[{"type": "tool_result", "content": [{"type": "x_url"}]}])
    assert input_bound([nested], {}) - input_bound([plain_nested], {}) == IMAGE_TOKENS + (
        len("image_url") - len("x_url")
    )


@pytest.mark.parametrize(
    "error",
    [
        ModelAuthenticationError("x"),
        ModelPermissionDeniedError("x"),
        ModelInvalidRequestError("x"),
        ModelNotFoundError("x"),
        ModelRateLimitError("x"),
        ContextOverflowError("x"),
    ],
)
async def test_each_rejection_settles_failed_at_zero(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession, error: Exception
) -> None:
    g, run = guard(), uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=run)
    await g.on_llm_error(error, run_id=run)
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("failed", 0)
    assert await total(db_session) == 0


@pytest.mark.parametrize("error", [ModelAPIError("x"), ModelTimeoutError("x"), KeyError("x")])
async def test_every_other_failure_settles_failed_at_the_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession, error: Exception
) -> None:
    g, run = guard(), uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=run)
    await g.on_llm_error(error, run_id=run)
    [row] = await rows(db_session)
    assert row.status == "failed"
    assert row.cost_micro_usd == row.reserved_micro_usd > 0
    assert await total(db_session) == row.reserved_micro_usd


async def test_a_rejection_after_streaming_started_keeps_the_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    g, streamed, plain = guard(), uuid.uuid4(), uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=streamed)
    await g.on_llm_new_token("Fo", run_id=streamed)
    await g.on_llm_error(ModelRateLimitError("x"), run_id=streamed)
    # The streaming mark belongs to its own call: the same guard, a different call, the same
    # rejection and no tokens streamed settles at zero. (The same run id cannot be used twice:
    # the call row's primary key is the run id.)
    await g.on_chat_model_start({}, [HELLO], run_id=plain)
    await g.on_llm_error(ModelRateLimitError("x"), run_id=plain)
    by_id = {r.id: r for r in await rows(db_session)}
    assert by_id[streamed].cost_micro_usd == by_id[streamed].reserved_micro_usd > 0
    assert by_id[plain].cost_micro_usd == 0
    assert await total(db_session) == by_id[streamed].reserved_micro_usd


async def test_cost_from_cache_and_reasoning_details_is_summed_then_rounded_up_once(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    g, run = guard(), uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=run)
    usage: Any = {
        "input_tokens": 1_001,
        "output_tokens": 200,
        "total_tokens": 1_201,
        "input_token_details": {
            "cache_read": 601,
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 100,
        },
        "output_token_details": {"reasoning": 50},
    }
    await g.on_llm_end(end_of(usage), run_id=run)
    [row] = await rows(db_session)
    # Plain 1,001 - 601 - 100 - 100 = 200 x 2 = 400; 601 x 0.1 = 60.1; 100 x 2.5 = 250;
    # 100 x 4 = 400; 200 output x 10 = 2,000 (reasoning is inside the output). Total 3,110.1,
    # rounded up once: 3,111.
    assert (row.status, row.cost_micro_usd) == ("settled", 3_111)
    assert (row.input_tokens, row.output_tokens) == (1_001, 200)
    assert (row.reasoning_tokens, row.cache_read_tokens, row.cache_write_tokens) == (
        50,
        601,
        200,
    )
    assert await total(db_session) == 3_111


async def test_missing_usage_logs_a_warning_and_keeps_the_reservation(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    g, run = guard(), uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=run)
    with capture_logs() as logs:
        await g.on_llm_end(end_of(None), run_id=run)
    [warning] = [log for log in logs if log["event"] == "llm_usage_missing"]
    assert (warning["model"], warning["call_id"]) == (MODEL, str(run))
    [row] = await rows(db_session)
    assert (row.status, row.input_tokens) == ("settled", None)
    assert row.cost_micro_usd == row.reserved_micro_usd


async def test_a_settlement_that_fails_does_not_fail_the_call_and_leaves_the_reservation(
    sessions: async_sessionmaker[AsyncSession], db_session: AsyncSession
) -> None:
    uses = {"count": 0}

    def flaky() -> async_sessionmaker[AsyncSession]:
        uses["count"] += 1
        if uses["count"] > 1:  # the first use reserves; the second (the settlement) breaks
            raise RuntimeError("database went away")
        return sessions

    g = BudgetGuard(
        model=MODEL,
        price=PRICE,
        cap_micro_usd=10**9,
        max_tokens=500,
        sessionmaker=flaky,
        today=lambda: DAY,
    )
    run = uuid.uuid4()
    await g.on_chat_model_start({}, [HELLO], run_id=run)
    with capture_logs() as logs:
        await g.on_llm_end(
            end_of({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}), run_id=run
        )  # does not raise
    assert [log["event"] for log in logs if log["log_level"] == "error"] == ["llm_settle_failed"]
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("reserved", None)
    assert await total(db_session) == row.reserved_micro_usd


async def test_a_refused_reservation_logs_which_scope_refused(
    guard: Callable[..., BudgetGuard],
) -> None:
    with capture_logs() as logs, pytest.raises(BudgetExhausted):
        await guard(cap=1).on_chat_model_start({}, [HELLO], run_id=uuid.uuid4())
    [refusal] = [log for log in logs if log["event"] == "llm_budget_exhausted"]
    assert (refusal["scope"], refusal["model"]) == ("day", MODEL)
    assert refusal["reservation_micro_usd"] > 1


# --- the spend report's grouping ----------------------------------------


async def seed(
    session: AsyncSession,
    day: date,
    model: str,
    prompt: str | None,
    status: str | None,
    cost: int,
    latency: int,
    tokens: tuple[int, int] = (10, 5),
) -> None:
    """One call row; status None leaves it reserved."""
    call_id = uuid.uuid4()
    assert await reserve(
        session, call_id=call_id, day=day, model=model, amount=cost, cap=10**9, prompt=prompt
    )
    if status:
        usage: Any = {
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
            "output_token_details": {"reasoning": 2},
        }
        await settle(
            session, call_id=call_id, status=status, cost=cost, usage=usage, latency_ms=latency
        )


async def test_the_report_groups_sums_and_orders_by_the_chosen_columns(
    db_session: AsyncSession,
) -> None:
    d1, d2 = date(2026, 3, 1), date(2026, 3, 2)
    await seed(db_session, d1, "m-a", "ask", "settled", 1_000_000, 10)
    await seed(db_session, d1, "m-a", "ask", "settled", 500_000, 20)
    await seed(db_session, d1, "m-a", "ask", "failed", 250_000, 30)
    await seed(db_session, d1, "m-b", None, "settled", 100, 5)
    await seed(db_session, d2, "m-a", "ask", "settled", 7, 1)
    await seed(db_session, d2, "m-a", "ask", None, 999, 0)  # still reserved: not in the report
    result = await report.spend(db_session, since=d1, by=["day", "model"])
    assert [r.group for r in result] == [
        ("2026-03-01", "m-a"),
        ("2026-03-01", "m-b"),
        ("2026-03-02", "m-a"),
    ]
    first = result[0]
    # Three calls, one failed; cost 1,000,000 + 500,000 + 250,000 = 1,750,000 micro-USD = $1.75;
    # tokens 3 x 10 in, 3 x 5 out, 3 x 2 reasoning.
    assert (first.calls, first.failed, first.cost_usd) == (3, 1, 1.75)
    assert (first.input_tokens, first.output_tokens, first.reasoning_tokens) == (30, 15, 6)
    # Latencies 10, 20, 30: the median is 20; p95 sits 0.95 x (3 - 1) = 1.9 steps along, so
    # 20 + 0.9 x (30 - 20) = 29.
    assert (first.p50_latency_ms, first.p95_latency_ms) == (20, 29)
    # A single call: both percentiles are its latency.
    assert (result[1].calls, result[1].p50_latency_ms, result[1].p95_latency_ms) == (1, 5, 5)
    assert (result[2].calls, result[2].cost_usd) == (1, 0.000007)


async def test_the_report_labels_a_missing_prompt_none_and_honours_since(
    db_session: AsyncSession,
) -> None:
    await seed(db_session, date(2026, 4, 1), "m-a", None, "settled", 10, 1)
    await seed(db_session, date(2026, 4, 2), "m-a", "ask", "settled", 20, 1)
    by_prompt = await report.spend(db_session, since=date(2026, 4, 1), by=["prompt"])
    assert sorted(r.group for r in by_prompt) == [("ask",), ("none",)]
    later = await report.spend(db_session, since=date(2026, 4, 2), by=["prompt"])
    assert [(r.group, r.calls) for r in later] == [(("ask",), 1)]
    assert await report.spend(db_session, since=date(2026, 4, 3), by=["prompt"]) == []


# --- embeddings: provider errors through the public call --------------------------------


class FailingEmbeddingsApi:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def create(self, **kwargs: Any) -> Any:
        raise self.error


def provider_errors() -> list[tuple[Exception, type[ModelError], bool]]:
    """(what OpenAI's SDK raises, what the gateway raises, whether OpenAI rejected it unbilled)."""
    request = httpx2.Request("POST", "http://x")

    def status(code: int) -> Any:
        return httpx2.Response(code, request=request)

    return [
        (
            openai.AuthenticationError("S", response=status(401), body=None),
            ModelAuthenticationError,
            True,
        ),
        (
            openai.PermissionDeniedError("S", response=status(403), body=None),
            ModelPermissionDeniedError,
            True,
        ),
        (openai.NotFoundError("S", response=status(404), body=None), ModelNotFoundError, True),
        (openai.RateLimitError("S", response=status(429), body=None), ModelRateLimitError, True),
        (
            openai.BadRequestError("S", response=status(400), body=None),
            ModelInvalidRequestError,
            True,
        ),
        (openai.APITimeoutError(request=request), ModelTimeoutError, False),
        (openai.APIConnectionError(request=request), ModelConnectionError, False),
        (openai.InternalServerError("S", response=status(500), body=None), ModelAPIError, False),
        (openai.APIError("S", request=request, body=None), ModelAPIError, False),
    ]


@pytest.mark.parametrize("case", range(9))
async def test_an_openai_embedding_error_becomes_the_model_error_family_and_settles_by_kind(
    sessions: async_sessionmaker[AsyncSession], db_session: AsyncSession, case: int
) -> None:
    # Private seam note: this pins embeddings._REJECTED and embeddings._as_model_error through the
    # public call. T4 moves them; keep this test and retarget it, do not delete it.
    source, target, unbilled = provider_errors()[case]
    api = FailingEmbeddingsApi(source)
    vectors = GatewayEmbeddings(
        model=EMBEDDING_MODEL,
        price=prices.EMBEDDING_PRICES[EMBEDDING_MODEL],
        client=SimpleNamespace(embeddings=api),
        cap_micro_usd=10**9,
        sessionmaker=lambda: sessions,
        today=lambda: DAY,
    )
    with pytest.raises(ModelError) as raised:
        await vectors.aembed_query("hello")
    # The type only, never the provider's message; the original error is the cause.
    assert type(raised.value) is target
    assert str(raised.value) == f"OpenAI embeddings: {type(source).__name__}"
    assert raised.value.__cause__ is source
    [row] = await rows(db_session)
    assert row.status == "failed"
    assert row.cost_micro_usd == (0 if unbilled else row.reserved_micro_usd)


async def test_an_error_that_is_not_openais_passes_through_unchanged(
    sessions: async_sessionmaker[AsyncSession], db_session: AsyncSession
) -> None:
    other = RuntimeError("not openai")
    vectors = GatewayEmbeddings(
        model=EMBEDDING_MODEL,
        price=prices.EMBEDDING_PRICES[EMBEDDING_MODEL],
        client=SimpleNamespace(embeddings=FailingEmbeddingsApi(other)),
        cap_micro_usd=10**9,
        sessionmaker=lambda: sessions,
        today=lambda: DAY,
    )
    with pytest.raises(RuntimeError) as raised:
        await vectors.aembed_query("hello")
    assert raised.value is other
    [row] = await rows(db_session)
    assert row.cost_micro_usd == row.reserved_micro_usd
