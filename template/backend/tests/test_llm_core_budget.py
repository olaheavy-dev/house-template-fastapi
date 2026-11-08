"""The framework-neutral request bounds (core/bounds.py) and metrics (core/metrics.py) of the
budget (#33 T3), called directly with neutral values, no LangChain involved. Numbers are worked
out by hand in the comments, never by the code under test.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

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
from app.llm.core.prices import Price, Rates
from app.llm.core.usage import Usage
from tests.test_llm_core import FRAMEWORKS

SOURCE = "https://example.com/pricing"
SONNET = Price(
    Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000), checked="2026-11-06", source=SOURCE
)


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


# --- the core stays framework-free ---------------------------------------------------------------


def test_importing_the_bounds_and_metrics_loads_no_framework() -> None:
    # A fresh interpreter: the test process has LangChain loaded already.
    code = (
        "import sys, app.llm.core.bounds, app.llm.core.metrics\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_the_bounds_module_holds_the_budget_constants() -> None:
    assert (bounds.INPUT_MARGIN_TOKENS, bounds.IMAGE_TOKENS) == (1_024, 6_000)
