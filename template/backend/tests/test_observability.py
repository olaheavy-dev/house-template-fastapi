"""What the app records about model calls: details and spend, OpenTelemetry metrics, and the spend
report."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm import prices, report
from app.llm.budget import BudgetGuard, LlmCall, reserve, settle
from app.llm.core.prices import cost
from app.llm.langchain.usage import from_langchain
from tests.models import CHAT_MODEL

pytestmark = pytest.mark.anyio

MODEL = CHAT_MODEL
DAY = date(2026, 1, 3)


def usage() -> UsageMetadata:
    return {
        "input_tokens": 1_000,
        "output_tokens": 300,
        "total_tokens": 1_300,
        "input_token_details": {"cache_read": 400},
        "output_token_details": {"reasoning": 120},
    }


@pytest.fixture(scope="session")
def metric_reader() -> InMemoryMetricReader:
    # The global meter provider can be set once per process; the llm metrics bind to it lazily.
    reader = InMemoryMetricReader()
    otel_metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    return reader


@pytest.fixture
def guard(db_session: AsyncSession) -> Callable[..., BudgetGuard]:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    return lambda cap=1_000_000: BudgetGuard(
        model=MODEL,
        price=prices.PRICES[MODEL],
        cap_micro_usd=cap,
        max_tokens=500,
        sessionmaker=lambda: sessions,
        today=lambda: DAY,
    )


async def call(guard: BudgetGuard, **metadata: str) -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage("ok", usage_metadata=usage())]), callbacks=[guard]
    )
    await model.ainvoke([HumanMessage("hi")], config={"metadata": metadata})


async def test_a_call_records_its_token_details_latency_and_tags(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    await call(guard(), feature="answer", user_ref="u_42", prompt="ask")
    row = await db_session.scalar(select(LlmCall).where(LlmCall.day == DAY))
    assert row is not None
    assert (row.reasoning_tokens, row.cache_read_tokens, row.cache_write_tokens) == (120, 400, 0)
    assert (row.feature, row.user_ref, row.prompt) == ("answer", "u_42", "ask")
    assert row.latency_ms is not None
    assert row.latency_ms >= 0


async def test_crossing_80_percent_of_the_cap_logs_one_warning(db_session: AsyncSession) -> None:
    def take(amount: int) -> Any:
        return reserve(
            db_session, call_id=uuid.uuid4(), day=DAY, model=MODEL, amount=amount, cap=1_000
        )

    with capture_logs() as logs:
        assert await take(700)  # 70%
        assert await take(150)  # 85%: crosses
        assert await take(100)  # 95%: already past
    warnings = [log for log in logs if log["event"] == "llm_budget_nearly_used"]
    assert len(warnings) == 1
    assert warnings[0]["committed_micro_usd"] == 850


async def test_the_warning_stays_once_a_day_when_settlements_lower_the_total(
    db_session: AsyncSession,
) -> None:
    # Codex review on PR 17: reservations settled below their amount lower the total, so a
    # later reservation can cross 80% again; the day must still warn once.
    with capture_logs() as logs:
        for _ in range(3):
            call_id = uuid.uuid4()
            assert await reserve(
                db_session, call_id=call_id, day=DAY, model=MODEL, amount=850, cap=1_000
            )
            await settle(db_session, call_id=call_id, status="settled", cost=10)
    assert [log["event"] for log in logs].count("llm_budget_nearly_used") == 1


def points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    assert data is not None
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


async def test_calls_are_counted_in_opentelemetry_by_model_prompt_and_feature(
    guard: Callable[..., BudgetGuard], metric_reader: InMemoryMetricReader
) -> None:
    await call(guard(), feature="digest", prompt="summarise")
    labels = {"model": MODEL, "prompt": "summarise", "feature": "digest"}
    [settled] = [
        p
        for p in points(metric_reader, "llm.calls")
        if p.attributes == {**labels, "outcome": "settled"}
    ]
    assert settled.value >= 1
    reasoning = [
        p
        for p in points(metric_reader, "llm.tokens")
        if p.attributes == {**labels, "kind": "reasoning"}
    ]
    assert reasoning[0].value >= 120
    assert any(p.attributes == labels for p in points(metric_reader, "llm.cost"))
    # No user ids in metric labels: they would make every user a separate series.
    assert all("user_ref" not in p.attributes for p in points(metric_reader, "llm.calls"))


async def test_the_spend_report_groups_settled_calls(
    guard: Callable[..., BudgetGuard], db_session: AsyncSession
) -> None:
    await call(guard(), feature="answer")
    await call(guard(), feature="answer")
    await call(guard(), feature="digest")
    rows = await report.spend(db_session, since=DAY, by=["feature"])
    by_feature = {row.group: row for row in rows}
    assert by_feature[("answer",)].calls == 2
    assert by_feature[("digest",)].reasoning_tokens == 120
    assert by_feature[("answer",)].cost_usd == pytest.approx(
        2 * cost(prices.PRICES[MODEL], from_langchain(usage())) / prices.MICRO_USD
    )
    table = report.render(rows, ["feature"])
    assert "answer" in table
    assert "total: $" in table
