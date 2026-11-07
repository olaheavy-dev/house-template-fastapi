"""Characterisation of the model gateway's public surface (#33): which names each app.llm module
offers, how provider failures map to outcomes and problem bodies, the metric names and attributes,
and the prompt version hash. The framework-neutral split must keep every one of these.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
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
from langgraph.errors import GraphRecursionError
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.db import Base
from app.llm import budget, embeddings, metrics, prices, report
from app.llm.errors import DEFAULT_RETRY_AFTER, OUTCOMES, Outcome, outcome_for
from app.llm.prompts import PROMPTS, Prompt, load_prompt

pytestmark = pytest.mark.anyio

# --- the import surface -------------------------------------------------------------------------

PUBLIC_NAMES: dict[str, tuple[str, ...]] = {
    "app.llm.core.usage": ("Usage",),
    "app.llm.core.prices": (
        "MICRO_USD",
        "Rates",
        "Price",
        "usd_rates",
        "cost",
        "ceiling",
        "token_details",
    ),
    "app.llm.core.registry": ("KEY_SETTINGS", "key_setting", "chat_price", "embedding_price"),
    "app.llm.langchain.usage": ("from_langchain",),
    "app.llm.prices": (
        "MICRO_USD",
        "Rates",
        "Price",
        "PRICES",
        "EMBEDDING_PRICES",
        "cost",
        "ceiling",
        "token_details",
    ),
    "app.llm.errors": (
        "Outcome",
        "OUTCOMES",
        "outcome_for",
        "install_model_error_handlers",
        "DEFAULT_RETRY_AFTER",
    ),
    "app.llm.budget": (
        "LlmSpendDay",
        "LlmSpendUserDay",
        "LlmCall",
        "BudgetExhausted",
        "UserBudgetExhausted",
        "PromptTooLong",
        "UnboundedMedia",
        "UnpricedRequest",
        "BudgetGuard",
        "reserve",
        "reserve_for",
        "settle",
        "expire_stale",
        "utc_today",
        "input_bound",
        "check_priced",
        "INPUT_MARGIN_TOKENS",
        "WARN_SHARE",
        "IMAGE_TOKENS",
    ),
    "app.llm.metrics": (
        "calls",
        "tokens",
        "cost",
        "latency",
        "labels",
        "record_refusal",
        "record_call",
    ),
    "app.llm.prompts": ("PROMPTS", "Prompt", "load_prompt"),
    "app.llm.embeddings": ("GatewayEmbeddings", "openai_client"),
    "app.llm.gateway": ("Gateway", "GatewayDep", "get_gateway", "key_setting", "KEY_SETTINGS"),
    "app.llm.checkpoints": (
        "MIGRATIONS",
        "TABLES",
        "apply_migrations",
        "migration_sql",
        "open_checkpointer",
    ),
    "app.llm.report": (
        "GROUPS",
        "SpendRow",
        "spend",
        "render",
        "stale",
        "collect",
        "main",
        "STALE_AFTER",
    ),
}


# Names that moved to the framework-neutral core (#33 T1). Their old path still answers, with a
# DeprecationWarning, until 0.8.0: test_llm_shims.py checks it gives the same object.
MOVED: dict[tuple[str, str], str] = {
    ("app.llm.prices", "cost"): "app.llm.core.prices",
    ("app.llm.prices", "ceiling"): "app.llm.core.prices",
    ("app.llm.prices", "token_details"): "app.llm.core.prices",
    ("app.llm.gateway", "KEY_SETTINGS"): "app.llm.core.registry",
    ("app.llm.gateway", "key_setting"): "app.llm.core.registry",
}


@pytest.mark.parametrize("module", sorted(PUBLIC_NAMES))
def test_every_public_name_is_importable_from_its_module(module: str) -> None:
    loaded = importlib.import_module(module)
    for name in PUBLIC_NAMES[module]:
        if (module, name) in MOVED:
            continue
        assert hasattr(loaded, name), f"{module}.{name}"
        # Resolving the name twice gives the same object (a shim must not rebuild it).
        assert getattr(loaded, name) is getattr(importlib.import_module(module), name)


def attr(module: object, name: str) -> Any:
    return getattr(module, name)


def test_names_imported_from_one_module_into_another_are_the_same_object() -> None:
    # Mapped SQLAlchemy classes in particular must exist once.
    assert attr(report, "LlmCall") is budget.LlmCall
    assert attr(report, "expire_stale") is budget.expire_stale
    assert attr(report, "MICRO_USD") is prices.MICRO_USD
    for name in (
        "BudgetExhausted",
        "UserBudgetExhausted",
        "reserve_for",
        "settle",
        "INPUT_MARGIN_TOKENS",
    ):
        assert attr(embeddings, name) is getattr(budget, name)


def test_the_mapped_table_and_column_names_are_unchanged() -> None:
    tables = Base.metadata.tables
    assert {"llm_call", "llm_spend_day", "llm_spend_user_day"} <= set(tables)
    assert budget.LlmCall.__tablename__ == "llm_call"
    assert budget.LlmSpendDay.__tablename__ == "llm_spend_day"
    assert budget.LlmSpendUserDay.__tablename__ == "llm_spend_user_day"
    assert set(tables["llm_call"].columns.keys()) == {
        "id",
        "day",
        "created_at",
        "settled_at",
        "model",
        "prompt",
        "prompt_sha",
        "status",
        "reserved_micro_usd",
        "cost_micro_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "latency_ms",
        "feature",
        "user_ref",
        "user_reserved",
    }
    assert set(tables["llm_spend_day"].columns.keys()) == {"day", "committed_micro_usd", "warned"}
    assert set(tables["llm_spend_user_day"].columns.keys()) == {
        "day",
        "user_ref",
        "committed_micro_usd",
    }


def test_the_budget_constants() -> None:
    assert budget.INPUT_MARGIN_TOKENS == 1024
    assert budget.IMAGE_TOKENS == 6_000
    assert budget.WARN_SHARE == 0.8


# --- failure kinds, outcomes and problem bodies --------------------------------------------------

BUSY = "The language model is busy. Try again shortly."
UNAVAILABLE = "The language model isn't available."


def test_the_outcome_table_is_in_this_order_with_these_answers() -> None:
    expected = (
        (ModelRateLimitError, Outcome(503, BUSY, True)),
        (ModelTimeoutError, Outcome(504, "The language model took too long to answer.", False)),
        (
            ContextOverflowError,
            Outcome(422, "The input is too long for the language model.", False),
        ),
        (ModelConnectionError, Outcome(502, "The language model can't be reached.", True)),
        (ModelAPIError, Outcome(502, "The language model failed to answer.", True)),
        (ModelAuthenticationError, Outcome(503, UNAVAILABLE, False)),
        (ModelPermissionDeniedError, Outcome(503, UNAVAILABLE, False)),
        (ModelNotFoundError, Outcome(503, UNAVAILABLE, False)),
    )
    assert expected == OUTCOMES
    assert DEFAULT_RETRY_AFTER == "30"


def test_outcome_for_an_invalid_request_is_none_and_a_subclass_matches_its_parent() -> None:
    assert outcome_for(ModelInvalidRequestError("bug")) is None
    assert outcome_for(ModelError("plain")) is None

    class Slow(ModelTimeoutError):
        pass

    assert outcome_for(Slow("late")) == Outcome(504, "The language model took too long to answer.")


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (ModelRateLimitError("x"), 503, BUSY),
        (ModelTimeoutError("x"), 504, "The language model took too long to answer."),
        (ContextOverflowError("x"), 422, "The input is too long for the language model."),
        (ModelConnectionError("x"), 502, "The language model can't be reached."),
        (ModelAPIError("x"), 502, "The language model failed to answer."),
        (ModelAuthenticationError("x"), 503, UNAVAILABLE),
        (ModelPermissionDeniedError("x"), 503, UNAVAILABLE),
        (ModelNotFoundError("x"), 503, UNAVAILABLE),
    ],
)
async def test_the_problem_body_for_each_provider_failure(
    app: FastAPI, client: httpx.AsyncClient, error: ModelError, status: int, detail: str
) -> None:
    @app.get("/model")
    async def model() -> None:
        raise error

    response = await client.get("/model")
    assert response.status_code == status
    # The title is Python's HTTPStatus phrase, which differs by version for 422.
    assert (
        response.json()["title"]
        in {
            503: {"Service Unavailable"},
            504: {"Gateway Timeout"},
            502: {"Bad Gateway"},
            422: {"Unprocessable Entity", "Unprocessable Content"},
        }[status]
    )
    assert response.json() == {
        "type": "about:blank",
        "title": response.json()["title"],
        "status": status,
        "detail": detail,
        "instance": "/model",
    }


async def test_a_retry_after_is_sent_only_when_the_outcome_says_retry(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    # Passing the provider's value on is test_llm_errors.py's. Here: a provider header is ignored
    # on an outcome that is not retryable (504, 503 auth) and, when retryable, a missing or empty
    # header falls back to 30.
    def with_header(error: ModelError, headers: dict[str, str]) -> ModelError:
        error.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
        return error

    errors_by_path = {
        "/timeout": with_header(ModelTimeoutError("x"), {"retry-after": "9"}),
        "/auth": with_header(ModelAuthenticationError("x"), {"retry-after": "9"}),
        "/rate-empty": with_header(ModelRateLimitError("x"), {"retry-after": ""}),
        "/rate-none": with_header(ModelRateLimitError("x"), {}),
        "/rate-bare": ModelRateLimitError("x"),
    }

    def add_route(path: str, error: ModelError) -> None:
        @app.get(path)
        async def route() -> None:
            raise error

    for path, error in errors_by_path.items():
        add_route(path, error)

    sent = {path: (await client.get(path)).headers.get("Retry-After") for path in errors_by_path}
    assert sent == {
        "/timeout": None,
        "/auth": None,
        "/rate-empty": "30",
        "/rate-none": "30",
        "/rate-bare": "30",
    }


async def test_an_unmapped_model_error_is_a_plain_500_problem_body(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/bug")
    async def bug() -> None:
        raise ModelInvalidRequestError("unknown parameter")

    response = await client.get("/bug")
    assert response.status_code == 500
    assert response.json() == {
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": 500,
        "instance": "/bug",
    }


async def test_the_agent_step_limit_is_a_422_problem(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/loop")
    async def loop() -> None:
        raise GraphRecursionError("recursion")

    response = await client.get("/loop")
    assert response.status_code == 422
    assert response.json() == {
        "type": "about:blank",
        "title": response.json()["title"],
        "status": 422,
        "detail": "The assistant couldn't finish this within its step limit.",
        "instance": "/loop",
    }


def test_budget_refusals_carry_their_status_detail_and_retry_after() -> None:
    midday = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    day = budget.BudgetExhausted(now=midday)
    assert (day.status, day.detail) == (
        503,
        "The daily model budget is used up. Try again tomorrow (UTC).",
    )
    assert day.headers == {"Retry-After": "43200"}  # 12 hours to midnight x 3,600 seconds
    user = budget.UserBudgetExhausted(now=midday)
    assert (user.status, user.detail) == (
        429,
        "Your daily model budget is used up. Try again tomorrow (UTC).",
    )
    assert user.headers == {"Retry-After": "43200"}
    # Less than one second left is still told to wait one second.
    last = datetime(2026, 1, 2, 23, 59, 59, 500_000, tzinfo=UTC)
    assert budget.BudgetExhausted(now=last).headers == {"Retry-After": "1"}
    assert budget.UserBudgetExhausted(now=last).headers == {"Retry-After": "1"}
    too_long = budget.PromptTooLong()
    assert (too_long.status, too_long.detail) == (
        422,
        "The input is too long for the language model.",
    )
    media = budget.UnboundedMedia("audio")
    assert (media.status, media.kind) == (422, "audio")
    assert media.detail == "This kind of attachment can't be sent to the language model."


def test_embedding_batches_are_bounded_by_count_and_bytes() -> None:
    assert embeddings.GatewayEmbeddings.MAX_INPUTS == 2048
    assert embeddings.GatewayEmbeddings.MAX_BYTES == 300_000 - 1024


# --- metrics --------------------------------------------------------------------------------------


def test_metric_labels_default_to_none() -> None:
    assert metrics.labels("m") == {"model": "m", "prompt": "none", "feature": "none"}
    assert metrics.labels("m", "", "") == {"model": "m", "prompt": "none", "feature": "none"}
    assert metrics.labels("m", "p", "f") == {"model": "m", "prompt": "p", "feature": "f"}


# The module that CREATES the instruments (calls, tokens, cost, latency). The fixture reloads this
# one. At T3 the creation moves to the core metrics module: change this one line.
METRICS_MODULE = metrics


@pytest.fixture
def local_metrics(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryMetricReader]:
    """The llm instruments created again on a private meter provider, so these tests read their
    own data and never touch the process-wide provider (it can be set once, and test_observability
    sets it). The instruments are restored afterwards. It must reload the module that creates the
    instruments (METRICS_MODULE), not a shim that re-exports them."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(otel_metrics, "get_meter", provider.get_meter)
    importlib.reload(METRICS_MODULE)
    yield reader
    monkeypatch.undo()
    importlib.reload(METRICS_MODULE)


def collect(reader: InMemoryMetricReader, model: str) -> dict[tuple[str, str, str], list[Any]]:
    """{(name, unit, description): [(attributes, value)]} for the points of one model label."""
    data = reader.get_metrics_data()
    assert data is not None
    found: dict[tuple[str, str, str], list[Any]] = {}
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                for point in metric.data.data_points:
                    if (point.attributes or {}).get("model") == model:
                        # Counters have a value; a histogram has a sum of what it recorded.
                        value = point.value if hasattr(point, "value") else point.sum
                        key = (metric.name, metric.unit or "", metric.description or "")
                        found.setdefault(key, []).append((dict(point.attributes or {}), value))
    return found


def test_a_settled_call_records_these_instruments_and_attributes(
    local_metrics: InMemoryMetricReader,
) -> None:
    labels = metrics.labels("pin-model-settled", "pin-prompt", "pin-feature")
    usage: Any = {
        "input_tokens": 1_000,
        "output_tokens": 300,
        "total_tokens": 1_300,
        "input_token_details": {"cache_read": 400, "cache_creation": 5},
        "output_token_details": {"reasoning": 120},
    }
    metrics.record_call(labels, "settled", 77, usage, 42)
    found = collect(local_metrics, "pin-model-settled")
    assert found[("llm.calls", "{call}", "Model calls by outcome")] == [
        ({**labels, "outcome": "settled"}, 1)
    ]
    assert found[("llm.cost", "{micro_usd}", "Settled model spend in micro-USD")] == [(labels, 77)]
    token_points = found[("llm.tokens", "{token}", "Tokens by kind")]
    assert sorted((a["kind"], v) for a, v in token_points) == [
        ("cache_read", 400),
        ("cache_write", 5),
        ("input", 1_000),
        ("output", 300),
        ("reasoning", 120),
    ]
    assert all({k: v for k, v in a.items() if k != "kind"} == labels for a, _ in token_points)
    [latency] = found[("llm.latency", "ms", "Time from a model call's start to its end")]
    assert latency[0] == {**labels, "outcome": "settled"}


def test_a_failed_call_without_cost_or_usage_records_only_the_count_and_latency(
    local_metrics: InMemoryMetricReader,
) -> None:
    labels = metrics.labels("pin-model-failed")
    metrics.record_call(labels, "failed", 0, None, 0)  # zero cost is not recorded; latency 0 is
    found = collect(local_metrics, "pin-model-failed")
    assert sorted(name for name, _, _ in found) == ["llm.calls", "llm.latency"]
    only = metrics.labels("pin-model-nolatency")
    metrics.record_call(only, "failed", None, None, None)
    assert [name for name, _, _ in collect(local_metrics, "pin-model-nolatency")] == ["llm.calls"]


def test_zero_token_kinds_are_not_recorded_and_a_refusal_counts_as_refused(
    local_metrics: InMemoryMetricReader,
) -> None:
    labels = metrics.labels("pin-model-zero")
    usage: Any = {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
    metrics.record_call(labels, "settled", None, usage, None)
    found = collect(local_metrics, "pin-model-zero")
    assert len([k for k in found if k[0] == "llm.tokens"]) == 1
    assert [(a["kind"], v) for a, v in found[("llm.tokens", "{token}", "Tokens by kind")]] == [
        ("input", 10)
    ]
    refused = metrics.labels("pin-model-refused", feature="f")
    metrics.record_refusal(refused)
    [(attrs, value)] = collect(local_metrics, "pin-model-refused")[
        ("llm.calls", "{call}", "Model calls by outcome")
    ]
    assert (attrs, value) == ({**refused, "outcome": "refused"}, 1)


# --- prompts --------------------------------------------------------------------------------------


def test_the_prompt_version_is_the_first_12_hex_of_the_sha256_of_the_text() -> None:
    # printf %s 'Say hello.' | shasum -a 256
    #   c8e2c1437abb87b67330d0dddbd1de9a179ca6be207497f14873894c26e7d742  -> first 12: c8e2c1437abb
    assert Prompt(name="greet", text="Say hello.").sha == "c8e2c1437abb"
    # The same text with a trailing newline is a different version (nothing is stripped):
    # printf 'Say hello.\n' | shasum -a 256 -> 3d023121b458...
    assert Prompt(name="greet", text="Say hello.\n").sha == "3d023121b458"
    # Hashed as UTF-8: printf %s 'Café ☕' | shasum -a 256 -> 0c3eecb944c2...
    assert Prompt(name="cafe", text="Café ☕").sha == "0c3eecb944c2"


def test_prompt_config_is_the_run_name_the_metadata_and_the_tag() -> None:
    assert Prompt(name="greet", text="Say hello.").config() == {
        "run_name": "greet",
        "metadata": {"prompt": "greet", "prompt_sha": "c8e2c1437abb"},
        "tags": ["prompt:greet@c8e2c1437abb"],
    }


def test_load_prompt_reads_the_markdown_file_unstripped_and_caches_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "ask.md").write_text("Answer.\n", encoding="utf-8")
    first = load_prompt("ask", tmp_path)
    assert (first.name, first.text) == ("ask", "Answer.\n")
    (tmp_path / "ask.md").write_text("Changed.", encoding="utf-8")
    assert load_prompt("ask", tmp_path) is first  # cached for the life of the process


def test_load_prompt_refuses_a_nested_name_and_a_missing_file(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.md").write_text("x")
    with pytest.raises(ValueError, match="not a prompt name"):
        load_prompt("sub/x", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_prompt("absent", tmp_path)


def test_the_prompts_directory_is_app_prompts() -> None:
    expected = Path(__file__).resolve().parents[1] / "src" / "app" / "prompts"
    assert expected == PROMPTS


# --- report rendering (no database) ---------------------------------------------------------------


def test_the_report_renders_a_fixed_table() -> None:
    rows = [
        report.SpendRow(("2026-01-02", "m1"), 3, 1, 0.000123, 1_000, 200, 50, 12, 30),
        report.SpendRow(("2026-01-02", "none"), 10, 0, 1.5, 5, 6, 0, None, None),
    ]
    assert report.render(rows, ["day", "model"]) == (
        "day         model  calls  failed  cost USD  in    out  reasoning  p50 ms  p95 ms\n"
        "----------  -----  -----  ------  --------  ----  ---  ---------  ------  ------\n"
        "2026-01-02  m1     3      1       0.000123  1000  200  50         12      30    \n"
        "2026-01-02  none   10     0       1.500000  5     6    0          -       -     \n"
        "\n"
        "total: $1.500123 over 13 calls"
    )
    assert report.GROUPS.keys() == {"day", "model", "prompt", "feature"}
