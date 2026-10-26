"""OpenTelemetry metrics for model calls, beside FastAPI's own HTTP metrics.

Every settled call adds to these instruments, labelled by model, prompt and feature (bounded sets;
never a user id). Without a meter provider they cost nothing: OpenTelemetry's API is a no-op until
one is configured (FASTAPI_OTEL_AUTO_CONFIGURE with an OTLP endpoint, or a provider of your own).
The llm_call table stays the record of truth; these are for dashboards and alerts.
"""

from __future__ import annotations

from langchain_core.messages.ai import UsageMetadata
from opentelemetry import metrics

from app.llm.prices import token_details

_meter = metrics.get_meter("app.llm")
calls = _meter.create_counter("llm.calls", unit="{call}", description="Model calls by outcome")
tokens = _meter.create_counter("llm.tokens", unit="{token}", description="Tokens by kind")
cost = _meter.create_counter(
    "llm.cost", unit="{micro_usd}", description="Settled model spend in micro-USD"
)
latency = _meter.create_histogram(
    "llm.latency", unit="ms", description="Time from a model call's start to its end"
)


def labels(model: str, prompt: str | None = None, feature: str | None = None) -> dict[str, str]:
    return {"model": model, "prompt": prompt or "none", "feature": feature or "none"}


def record_refusal(attributes: dict[str, str]) -> None:
    calls.add(1, {**attributes, "outcome": "refused"})


def record_call(
    attributes: dict[str, str],
    status: str,
    cost_micro_usd: int | None,
    usage: UsageMetadata | None,
    latency_ms: int | None,
) -> None:
    calls.add(1, {**attributes, "outcome": status})
    if cost_micro_usd:
        cost.add(cost_micro_usd, attributes)
    if latency_ms is not None:
        latency.record(latency_ms, {**attributes, "outcome": status})
    if usage:
        counts = {"input": usage["input_tokens"], "output": usage["output_tokens"]}
        details = token_details(usage)
        counts |= {
            "reasoning": details["reasoning_tokens"] or 0,
            "cache_read": details["cache_read_tokens"] or 0,
            "cache_write": details["cache_write_tokens"] or 0,
        }
        for kind, count in counts.items():
            if count:
                tokens.add(count, {**attributes, "kind": kind})
