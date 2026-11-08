"""The framework-neutral budget (#33 T3): the request bounds (core/bounds.py), the Meter an adapter
drives (core/budget.py) and the metrics it records (core/metrics.py), called directly with neutral
values, no LangChain involved. Money values are worked out by hand in the comments, never by the
code under test (a rate in USD per million tokens is the same number in micro-USD per token).

Shapes: SONNET input 2, output 10, cache read 0.1, write 5m 2.5, write 1h 4 (highest input 4);
FRACTIONS input 0.1, output 0.5, long write 0.25 (highest input 0.25).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import pytest
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core import bounds, metrics
from app.llm.core.bounds import (
    IMAGE_TOKENS,
    INPUT_MARGIN_TOKENS,
    PromptTooLong,
    UnboundedMedia,
    check_prompt_size,
    count_images,
    output_bound,
    request_bound,
)
from app.llm.core.budget import (
    BudgetExhausted,
    CallTags,
    LlmCall,
    LlmSpendDay,
    LlmSpendUserDay,
    Meter,
    UserBudgetExhausted,
)
from app.llm.core.errors import FailureKind
from app.llm.core.prices import Price, Rates
from app.llm.core.usage import Usage
from tests.test_llm_core import FRAMEWORKS

pytestmark = pytest.mark.anyio

SOURCE = "https://example.com/pricing"
SONNET = Price(
    Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000), checked="2026-11-06", source=SOURCE
)
FRACTIONS = Price(Rates(100_000, 500_000, 0, 0, 250_000), checked="2026-11-06", source=SOURCE)
DAY = date(2026, 2, 7)
MODEL = "anthropic:pin-meter"


# --- bounds ---------------------------------------------------------------------------------------


def test_the_request_bound_is_the_payloads_utf8_bytes_plus_images_plus_the_margin() -> None:
    assert (INPUT_MARGIN_TOKENS, IMAGE_TOKENS) == (1_024, 6_000)
    # json.dumps([]) is the 2 bytes "[]": 2 + 1,024 = 1,026.
    assert request_bound([], []) == 1_026
    # '{"a": "é"}' is 10 characters, 11 UTF-8 bytes (é is two): 11 + 1,024 = 1,035.
    assert request_bound({"a": "é"}, []) == 1_035
    # '"x"' is 3 bytes, two images 2 x 6,000 = 12,000: 3 + 12,000 + 1,024 = 13,027.
    assert request_bound("x", [{"type": "image_url"}, {"type": "input_image"}]) == 13_027


def test_images_are_counted_anywhere_in_the_content_and_media_is_refused() -> None:
    content = [
        "plain text",
        {"type": "image"},
        {"type": "tool_result", "content": [{"type": "image_url", "url": "u"}]},
        {"type": "text", "text": "image"},  # the word is not a block type
    ]
    assert count_images(content) == 2
    assert count_images("a string") == 0
    for kind in ("document", "file", "input_file", "audio", "input_audio", "video"):
        with pytest.raises(UnboundedMedia) as refused:
            count_images([{"type": "text"}, {"wrapper": {"type": kind}}])
        assert refused.value.kind == kind


def test_the_output_bound_is_the_largest_integer_limit() -> None:
    assert output_bound(500, {}) == 500
    assert output_bound(500, {"max_tokens": 700}) == 700
    assert output_bound(500, {"max_tokens": 700, "max_completion_tokens": 900}) == 900
    assert output_bound(500, {"max_tokens": 100}) == 500  # smaller: the gateway's stays
    assert output_bound(500, {"max_tokens": "9000", "max_completion_tokens": None}) == 500


def test_the_size_bound_allows_the_limit_and_refuses_one_more() -> None:
    limited = Price(SONNET.rates, checked="2026-11-06", source=SOURCE, max_prompt_tokens=100)
    check_prompt_size(limited, 100)
    with pytest.raises(PromptTooLong):
        check_prompt_size(limited, 101)
    check_prompt_size(SONNET, 10**9)  # no limit
    # A limit of 0 has always meant no limit (the check is on its truth value).
    zero = Price(SONNET.rates, checked="2026-11-06", source=SOURCE, max_prompt_tokens=0)
    check_prompt_size(zero, 10**9)


# --- the Meter ------------------------------------------------------------------------------------


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def meter(sessions: async_sessionmaker[AsyncSession]) -> Callable[..., Meter]:
    def make(price: Price = SONNET, cap: int = 10**9, **options: Any) -> Meter:
        return Meter(
            model=MODEL,
            price=price,
            cap_micro_usd=cap,
            sessionmaker=lambda: sessions,
            today=lambda: DAY,
            **options,
        )

    return make


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def total(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def test_a_call_reserves_its_ceiling_then_settles_at_its_cost(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    clock = iter([100.0, 100.25])
    m, call = meter(clock=lambda: next(clock)), uuid.uuid4()
    # Ceiling: 1,000 input x 4 (highest input rate) + 500 output x 10 = 9,000.
    assert await m.reserve(call, input_tokens=1_000, output_tokens=500) == 9_000
    assert await total(db_session) == 9_000
    await m.settle(call, Usage(input_tokens=1_000, output_tokens=500))
    [row] = await rows(db_session)
    # Cost: 1,000 x 2 + 500 x 10 = 7,000; the day moves by 7,000 - 9,000.
    assert (row.status, row.reserved_micro_usd, row.cost_micro_usd) == ("settled", 9_000, 7_000)
    assert (row.input_tokens, row.output_tokens) == (1_000, 500)
    # Details the Usage holds as 0 are recorded as 0, not unknown.
    assert (row.reasoning_tokens, row.cache_read_tokens, row.cache_write_tokens) == (0, 0, 0)
    assert row.latency_ms == 250  # (100.25 - 100.0) x 1,000
    assert await total(db_session) == 7_000


async def test_cost_and_ceiling_round_up_once_on_the_total(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m, call = meter(price=FRACTIONS), uuid.uuid4()
    # Ceiling: 3 x 0.25 + 1 x 0.5 = 1.25, rounded up once: 2 (per line: 1 + 1 = 2 as well).
    assert await m.reserve(call, input_tokens=3, output_tokens=1) == 2
    # Cost: 3 x 0.1 + 1 x 0.5 = 0.8, rounded up once: 1 (per line it would be 1 + 1 = 2).
    await m.settle(call, Usage(input_tokens=3, output_tokens=1))
    [row] = await rows(db_session)
    assert row.cost_micro_usd == 1
    assert await total(db_session) == 1


async def test_cache_and_reasoning_details_are_priced_and_recorded(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m, call = meter(), uuid.uuid4()
    await m.reserve(call, input_tokens=2_000, output_tokens=500)
    usage = Usage(
        input_tokens=1_001,
        output_tokens=200,
        reasoning_tokens=50,
        cache_read_tokens=601,
        cache_write_short_tokens=100,
        cache_write_long_tokens=60,
        cache_write_undivided_tokens=40,
    )
    await m.settle(call, usage)
    [row] = await rows(db_session)
    # Plain 1,001 - 601 - 100 - 60 - 40 = 200 x 2 = 400; 601 x 0.1 = 60.1; 100 x 2.5 = 250;
    # (60 + 40) x 4 = 400 (no lifetime: the longer one's rate); 200 output x 10 = 2,000.
    # 3,110.1 rounded up once: 3,111.
    assert row.cost_micro_usd == 3_111
    assert (row.reasoning_tokens, row.cache_read_tokens, row.cache_write_tokens) == (50, 601, 200)


async def test_a_call_without_usage_keeps_its_reservation_and_warns(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m, call = meter(), uuid.uuid4()
    await m.reserve(call, input_tokens=1_000, output_tokens=500)
    with capture_logs() as logs:
        await m.settle(call, None)
    [warning] = [log for log in logs if log["event"] == "llm_usage_missing"]
    assert (warning["model"], warning["call_id"], warning["log_level"]) == (
        MODEL,
        str(call),
        "warning",
    )
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd, row.input_tokens, row.reasoning_tokens) == (
        "settled",
        9_000,
        None,
        None,
    )


# Written out, not read from core/errors.py REJECTED: the kinds a failed call is refunded for.
REFUNDED = {
    FailureKind.AUTHENTICATION,
    FailureKind.PERMISSION_DENIED,
    FailureKind.INVALID_REQUEST,
    FailureKind.MODEL_NOT_FOUND,
    FailureKind.RATE_LIMITED,
    FailureKind.CONTEXT_OVERFLOW,
}


@pytest.mark.parametrize("kind", [*FailureKind, None], ids=str)
@pytest.mark.parametrize("streamed", [False, True], ids=["not-streamed", "streamed"])
async def test_a_failed_call_is_refunded_only_for_a_rejection_that_streamed_nothing(
    meter: Callable[..., Meter],
    db_session: AsyncSession,
    kind: FailureKind | None,
    streamed: bool,
) -> None:
    m, call = meter(), uuid.uuid4()
    await m.reserve(call, input_tokens=1_000, output_tokens=500)  # 9,000, as above
    await m.failed(call, kind, streamed=streamed)
    [row] = await rows(db_session)
    refunded = kind in REFUNDED and not streamed
    assert (row.status, row.cost_micro_usd) == ("failed", 0 if refunded else 9_000)
    assert (row.input_tokens, row.reasoning_tokens) == (None, None)
    assert await total(db_session) == (0 if refunded else 9_000)


async def test_the_tags_are_recorded_with_the_call(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m, call = meter(), uuid.uuid4()
    tags = CallTags(prompt="ask", prompt_sha="abc123", feature="answer", user_ref="u_9")
    await m.reserve(call, input_tokens=10, output_tokens=10, tags=tags)
    [row] = await rows(db_session)
    assert (row.prompt, row.prompt_sha, row.feature, row.user_ref, row.user_reserved) == (
        "ask",
        "abc123",
        "answer",
        "u_9",
        False,  # no user cap set
    )


async def test_a_reservation_over_a_cap_is_refused_and_records_nothing(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    # The ceiling of (1,000, 500) is 9,000: a cap of 8,999 refuses it, 9,000 allows it.
    with capture_logs() as logs, pytest.raises(BudgetExhausted):
        await meter(cap=8_999).reserve(uuid.uuid4(), input_tokens=1_000, output_tokens=500)
    [refusal] = [log for log in logs if log["event"] == "llm_budget_exhausted"]
    assert (refusal["scope"], refusal["reservation_micro_usd"]) == ("day", 9_000)
    assert (await rows(db_session), await total(db_session)) == ([], None)

    user = meter(user_cap_micro_usd=8_999)
    with pytest.raises(UserBudgetExhausted) as refused:
        await user.reserve(
            uuid.uuid4(), input_tokens=1_000, output_tokens=500, tags=CallTags(user_ref="u_1")
        )
    assert refused.value.status == 429
    # Untagged, the same call is not the user's: only the day's cap applies.
    await user.reserve(uuid.uuid4(), input_tokens=1_000, output_tokens=500)
    assert await total(db_session) == 9_000
    assert await meter(cap=9_000).reserve(uuid.uuid4(), input_tokens=0, output_tokens=0) == 0


async def test_a_single_reservation_may_equal_the_cap_on_a_fresh_day(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    # The ceiling of (1,000, 500) is 9,000 (above): equal to the cap is allowed ("<= cap").
    assert (
        await meter(cap=9_000).reserve(uuid.uuid4(), input_tokens=1_000, output_tokens=500) == 9_000
    )
    assert await total(db_session) == 9_000


async def test_a_users_capped_call_moves_the_users_total_by_its_cost(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m, call = meter(user_cap_micro_usd=9_000), uuid.uuid4()
    await m.reserve(call, input_tokens=1_000, output_tokens=500, tags=CallTags(user_ref="u_2"))
    await m.settle(call, Usage(input_tokens=1_000, output_tokens=500))
    user = await db_session.scalar(
        select(LlmSpendUserDay.committed_micro_usd).where(LlmSpendUserDay.user_ref == "u_2")
    )
    assert (user, await total(db_session)) == (7_000, 7_000)  # 9,000 reserved, 7,000 cost


async def test_a_prompt_past_the_size_bound_is_refused_before_any_database_work(
    meter: Callable[..., Meter],
) -> None:
    def no_database() -> async_sessionmaker[AsyncSession]:
        raise AssertionError("reserved")

    limited = Price(SONNET.rates, checked="2026-11-06", source=SOURCE, max_prompt_tokens=999)
    m = Meter(model=MODEL, price=limited, cap_micro_usd=10**9, sessionmaker=no_database)
    with pytest.raises(PromptTooLong):
        await m.reserve(uuid.uuid4(), input_tokens=1_000, output_tokens=0)


async def test_settings_changed_after_construction_apply_to_the_next_call(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m = meter()
    m.cap = 8_999
    with pytest.raises(BudgetExhausted):
        await m.reserve(uuid.uuid4(), input_tokens=1_000, output_tokens=500)
    m.cap, m.price = 10**9, FRACTIONS
    assert await m.reserve(uuid.uuid4(), input_tokens=3, output_tokens=1) == 2  # as above


async def test_a_settlement_that_fails_is_logged_and_leaves_the_reservation(
    sessions: async_sessionmaker[AsyncSession], db_session: AsyncSession
) -> None:
    used: list[str] = []

    def once() -> async_sessionmaker[AsyncSession]:
        used.append("x")
        if len(used) > 1:  # the first use reserves; the second (the settlement) breaks
            raise RuntimeError("database went away")
        return sessions

    m, call = Meter(model=MODEL, price=SONNET, cap_micro_usd=10**9, sessionmaker=once), uuid.uuid4()
    m.today = lambda: DAY
    await m.reserve(call, input_tokens=1_000, output_tokens=500)
    with capture_logs() as logs:
        await m.settle(call, Usage(input_tokens=1, output_tokens=1))  # does not raise
    assert [log["event"] for log in logs if log["log_level"] == "error"] == ["llm_settle_failed"]
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("reserved", None)


async def test_ending_a_call_that_was_never_reserved_changes_nothing(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m = meter()
    await m.settle(uuid.uuid4(), Usage(input_tokens=1, output_tokens=1))
    await m.failed(uuid.uuid4(), FailureKind.RATE_LIMITED, streamed=False)
    assert (await rows(db_session), await total(db_session)) == ([], None)


# --- metrics --------------------------------------------------------------------------------------


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryMetricReader]:
    """The llm instruments made again on a private meter provider (as in test_llm_surface)."""
    found = InMemoryMetricReader()
    monkeypatch.setattr(otel_metrics, "get_meter", MeterProvider(metric_readers=[found]).get_meter)
    importlib.reload(metrics)
    yield found
    monkeypatch.undo()
    importlib.reload(metrics)


def points(reader: InMemoryMetricReader, model: str) -> dict[str, list[tuple[dict[str, Any], Any]]]:
    data = reader.get_metrics_data()
    assert data is not None
    found: dict[str, list[tuple[dict[str, Any], Any]]] = {}
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                for point in metric.data.data_points:
                    attributes = dict(point.attributes or {})
                    if attributes.get("model") == model:
                        value = point.value if hasattr(point, "value") else point.sum
                        found.setdefault(metric.name, []).append((attributes, value))
    return found


def test_record_call_counts_each_kind_of_token_from_the_usage(
    reader: InMemoryMetricReader,
) -> None:
    labels = metrics.labels("pin-core-usage", "p", "f")
    usage = Usage(
        input_tokens=10,
        output_tokens=4,
        reasoning_tokens=3,
        cache_read_tokens=2,
        cache_write_short_tokens=1,
        cache_write_undivided_tokens=1,
    )
    metrics.record_call(labels, "settled", 5, usage, 7)
    found = points(reader, "pin-core-usage")
    # cache_write is short + long + undivided: 1 + 0 + 1 = 2.
    assert sorted((a["kind"], v) for a, v in found["llm.tokens"]) == [
        ("cache_read", 2),
        ("cache_write", 2),
        ("input", 10),
        ("output", 4),
        ("reasoning", 3),
    ]
    assert found["llm.cost"] == [(labels, 5)]
    assert found["llm.calls"] == [({**labels, "outcome": "settled"}, 1)]


def test_a_zero_usage_records_no_tokens_and_none_records_no_tokens(
    reader: InMemoryMetricReader,
) -> None:
    metrics.record_call(
        metrics.labels("pin-core-zero"), "settled", 0, Usage(input_tokens=0, output_tokens=0), 0
    )
    assert sorted(points(reader, "pin-core-zero")) == ["llm.calls", "llm.latency"]
    metrics.record_call(metrics.labels("pin-core-none"), "failed", None, None, None)
    assert sorted(points(reader, "pin-core-none")) == ["llm.calls"]


async def test_a_metered_call_records_its_metrics_with_its_tags(
    reader: InMemoryMetricReader, meter: Callable[..., Meter]
) -> None:
    clock = iter([5.0, 5.5])
    m = Meter(
        model="pin-core-meter",
        price=SONNET,
        cap_micro_usd=10**9,
        sessionmaker=meter().sessionmaker,
        today=lambda: DAY,
        clock=lambda: next(clock),
    )
    call = uuid.uuid4()
    tags = CallTags(prompt="ask", feature="answer", user_ref="u_never_a_label")
    await m.reserve(call, input_tokens=1_000, output_tokens=500, tags=tags)
    await m.settle(call, Usage(input_tokens=1_000, output_tokens=500))
    found = points(reader, "pin-core-meter")
    labels = {"model": "pin-core-meter", "prompt": "ask", "feature": "answer"}
    assert found["llm.calls"] == [({**labels, "outcome": "settled"}, 1)]
    assert found["llm.cost"] == [(labels, 7_000)]
    assert found["llm.latency"] == [({**labels, "outcome": "settled"}, 500)]  # 0.5 s
    m.cap = 1
    with pytest.raises(BudgetExhausted):
        await m.reserve(uuid.uuid4(), input_tokens=1, output_tokens=1, tags=tags)
    refused = [
        v for a, v in points(reader, "pin-core-meter")["llm.calls"] if a["outcome"] == "refused"
    ]
    assert refused == [1]


# --- the core stays framework-free ---------------------------------------------------------------


def test_importing_the_budget_bounds_and_metrics_loads_no_framework() -> None:
    # A fresh interpreter: the test process has LangChain loaded already.
    code = (
        "import sys, app.llm.core.budget, app.llm.core.bounds, app.llm.core.metrics\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_the_bounds_module_holds_the_budget_constants() -> None:
    assert (bounds.INPUT_MARGIN_TOKENS, bounds.IMAGE_TOKENS) == (1_024, 6_000)
