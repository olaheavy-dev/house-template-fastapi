"""What redaction reports about itself (#78): counts and kinds only, never a value. The
`llm.redactions` metric by kind, and the warning for media that is not scanned (test 30)."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, NumberDataPoint
from structlog.testing import capture_logs

from app.llm.core import metrics
from tests import wire
from tests.wire import EMAIL, NINO, PHONE

pytestmark = pytest.mark.anyio

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryMetricReader]:
    """The llm instruments created again on a private meter provider (as in test_llm_surface)."""
    found = InMemoryMetricReader()
    monkeypatch.setattr(otel_metrics, "get_meter", MeterProvider(metric_readers=[found]).get_meter)
    importlib.reload(metrics)
    yield found
    monkeypatch.undo()
    importlib.reload(metrics)


def redaction_points(reader: InMemoryMetricReader) -> dict[tuple[str, str], list[Any]]:
    """{(name, unit): [(attributes, value)]} for the llm.redactions points."""
    data = reader.get_metrics_data()
    found: dict[tuple[str, str], list[Any]] = {}
    if data is None:  # nothing was recorded at all
        return found
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name == "llm.redactions":
                    for point in metric.data.data_points:
                        assert isinstance(point, NumberDataPoint)  # a counter
                        key = (metric.name, metric.unit or "")
                        found.setdefault(key, []).append(
                            (dict(point.attributes or {}), point.value)
                        )
    return found


async def test_the_metric_counts_replaced_values_by_kind_and_carries_no_other_label(
    reader: InMemoryMetricReader,
) -> None:
    _, model = wire.wrapped("openai")
    # Email twice and a phone once: the same email counts each time it is replaced.
    text = f"Mail {EMAIL} again {EMAIL}, call {PHONE}. Nothing else."
    with wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage(text)])
    points = redaction_points(reader)
    assert sorted(points) == [("llm.redactions", "{item}")]
    by_kind = {attrs["kind"]: value for attrs, value in points[("llm.redactions", "{item}")]}
    assert by_kind == {"EMAIL": 2, "PHONE": 1}
    for attrs, _ in points[("llm.redactions", "{item}")]:
        assert set(attrs) == {"kind"}  # a detector's name, never a value, a model or a user


async def test_a_request_with_nothing_to_redact_adds_no_point(
    reader: InMemoryMetricReader,
) -> None:
    _, model = wire.wrapped("openai")
    with wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage("What is a freehold?")])
    assert redaction_points(reader) == {}


async def test_streamed_calls_count_too(reader: InMemoryMetricReader) -> None:
    _, model = wire.wrapped("openai")
    with wire.captured(lambda _b: wire.openai_stream_text(["ok"])):
        _ = [c async for c in model.astream([HumanMessage(f"NI {NINO}")])]
    [(attrs, value)] = redaction_points(reader)[("llm.redactions", "{item}")]
    assert (attrs, value) == ({"kind": "NINO"}, 1)


def test_record_redactions_skips_kinds_with_no_values(reader: InMemoryMetricReader) -> None:
    metrics.record_redactions({"EMAIL": 0, "PHONE": 3})
    [(attrs, value)] = redaction_points(reader)[("llm.redactions", "{item}")]
    assert (attrs, value) == ({"kind": "PHONE"}, 3)


# --- images (test 30) ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_images_are_counted_as_unscanned_in_a_log_with_no_text(provider: str) -> None:
    _, model = wire.wrapped(provider)
    messages: list[BaseMessage] = [
        HumanMessage(content=[{"type": "text", "text": f"Who is {EMAIL}?"}, IMAGE, IMAGE]),
        # An image inside a tool result's blocks counts too.
        ToolMessage(content=[IMAGE], tool_call_id="call_1"),
    ]
    with capture_logs() as logs, wire.captured(lambda _b: wire.text_reply(provider, "ok")):
        await model.ainvoke(messages)
    [event] = [e for e in logs if e["event"] == "llm_unscanned_images"]
    assert event["count"] == 3
    assert set(event) == {"event", "log_level", "count"}  # a count, nothing else
    assert EMAIL not in repr(logs)
    assert "iVBORw0KGgo" not in repr(logs)


async def test_other_media_count_as_unscanned_and_text_only_requests_log_nothing() -> None:
    _, model = wire.wrapped("openai")
    file_block = {
        "type": "file",
        "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"},
    }
    with capture_logs() as logs, wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage(content=[file_block])])
        await model.ainvoke([HumanMessage("just text")])
    [event] = [e for e in logs if e["event"] == "llm_unscanned_images"]
    assert event["count"] == 1
