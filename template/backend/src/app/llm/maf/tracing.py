"""Tracing MAF model calls to LangSmith over OpenTelemetry (#35 PR 7, #78 part 4).

What is sent: MAF's own spans (`invoke_agent`, `chat`, `execute_tool`, the workflow spans; tracer
scope "agent_framework") and the gateway's `llm.call` span (scope "app.llm", one per model call,
around MAF's `chat` span), which carries the call's prompt name and hash, its feature and its
`llm_call` row id as LangSmith metadata (`langsmith.metadata.*`). Never message text: MAF writes
prompts, answers and tool arguments into spans only with its sensitive data capture on
(`gen_ai.input.messages` and the like), which is set explicitly off here and which the app refuses
to start with while redaction is on (app/settings.py). FastAPI's request spans are not sent to
LangSmith.

How: one OpenTelemetry TracerProvider for the process. If FastAPI's own telemetry (or the app)
already installed an SDK provider, the LangSmith processor is added to it; otherwise one is created
and installed. MAF's instrumentation is turned on with
`enable_instrumentation(enable_sensitive_data= ..., force=True)`, so the ENABLE_SENSITIVE_DATA
environment variable cannot switch capture on, and MAF's own provider setup
(`configure_otel_providers`, which replaces the global providers) is marked as done so that nothing
calls it on top. Spans go to LangSmith's OTLP/HTTP endpoint with the `x-api-key` header (and
`Langsmith-Project` when a project is set). The share of traces sent is
LANGSMITH_TRACING_SAMPLING_RATE, decided per trace. At shutdown the processor is flushed and closed
within a time limit: an unreachable endpoint must not hold the process (the founder's MAF lab saw
over a minute).

Sources: MAF observability (Python), https://learn.microsoft.com/en-us/agent-framework/agents/
observability; LangSmith and OpenTelemetry, https://docs.langchain.com/langsmith/trace-with-
opentelemetry (its MAF page uses an older MAF API).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from agent_framework.observability import OBSERVABILITY_SETTINGS, enable_instrumentation
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from app.settings import Settings

__all__ = [
    "LANGSMITH_OTEL_ENDPOINT",
    "SCOPE",
    "Tracing",
    "call_span",
    "configure",
    "filtered",
]

log = structlog.get_logger()

SCOPE = "app.llm"  # the tracer name of the gateway's own spans
MAF_SCOPE = "agent_framework"  # MAF's tracer name (its docs and source)
LANGSMITH_OTEL_ENDPOINT = "https://api.smith.langchain.com/otel"
SHUTDOWN_SECONDS = 5.0

_tracer = trace.get_tracer(SCOPE)


class _Filtered(SpanExporter):
    """Passes on only MAF's and the gateway's spans, and only for the sampled share of traces."""

    def __init__(self, inner: SpanExporter, rate: float) -> None:
        self.inner = inner
        # A trace id is 128 random bits: its low 64 bits compared with the rate decide the whole
        # trace, so a trace is sent whole or not at all.
        self.bound = int(rate * (1 << 64))

    def _keep(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
        if scope != MAF_SCOPE and scope != SCOPE:
            return False
        context = span.get_span_context()
        return context is not None and (context.trace_id & ((1 << 64) - 1)) < self.bound

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = [span for span in spans if self._keep(span)]
        return self.inner.export(kept) if kept else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.inner.force_flush(timeout_millis)


def filtered(exporter: SpanExporter, rate: float = 1.0) -> SpanExporter:
    """`exporter` behind the filter (a test seam: tests pass an in-memory exporter)."""
    return _Filtered(exporter, rate)


@dataclass
class Tracing:
    """What configure() set up; `shutdown()` at the end of the app's lifespan."""

    processor: BatchSpanProcessor | None = None

    def shutdown(self, timeout_seconds: float = SHUTDOWN_SECONDS) -> None:
        processor = self.processor
        if processor is None:
            return

        def close() -> None:
            processor.force_flush(int(timeout_seconds * 1000))
            processor.shutdown()  # type: ignore[no-untyped-call]  # unannotated in otel 1.45

        worker = threading.Thread(target=close, name="langsmith-span-shutdown", daemon=True)
        worker.start()
        worker.join(timeout_seconds)


def _provider() -> TracerProvider:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    provider = TracerProvider(shutdown_on_exit=False)
    trace.set_tracer_provider(provider)
    return provider


def _langsmith_exporter(settings: Settings) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert settings.langsmith_api_key is not None  # noqa: S101 - checked by the caller
    base = (settings.langsmith_endpoint or LANGSMITH_OTEL_ENDPOINT).rstrip("/")
    headers = {"x-api-key": settings.langsmith_api_key.get_secret_value()}
    if settings.langsmith_project:
        headers["Langsmith-Project"] = settings.langsmith_project
    return OTLPSpanExporter(endpoint=f"{base}/v1/traces", headers=headers, timeout=10)


def configure(settings: Settings, *, exporter: SpanExporter | None = None) -> Tracing:
    """Turn MAF's instrumentation on (sensitive data as the settings say, which the settings only
    allow off while redaction is on), keep MAF from installing providers of its own, and send the
    spans to LangSmith when LANGSMITH_TRACING and LANGSMITH_API_KEY are set (never in tests).
    `exporter` replaces LangSmith's (a test seam)."""
    enable_instrumentation(enable_sensitive_data=settings.enable_sensitive_data, force=True)
    # Private to agent-framework-core 1.25.0 (the founder's lab does the same); a test pins it.
    OBSERVABILITY_SETTINGS._executed_setup = True
    if exporter is None:
        wanted = settings.langsmith_tracing and settings.app_env != "test"
        if not wanted:
            return Tracing()
        if settings.langsmith_api_key is None:
            log.warning("langsmith_tracing_without_key")
            return Tracing()
        exporter = _langsmith_exporter(settings)
    processor = BatchSpanProcessor(filtered(exporter, settings.langsmith_tracing_sampling_rate))
    _provider().add_span_processor(processor)
    log.info("llm_tracing", state="on", sampling_rate=settings.langsmith_tracing_sampling_rate)
    return Tracing(processor)


@contextmanager
def call_span(tags: Any) -> Iterator[trace.Span]:
    """The `llm.call` span of one model call, current while the call is made (MAF's `chat` span
    becomes its child). Its attributes are the call's tags as LangSmith metadata: the prompt's
    name and hash and the feature. Never text, never the user. The caller ends it (a streamed
    call ends after its stream)."""
    attributes: dict[str, str] = {"langsmith.span.kind": "chain"}
    for name in ("prompt", "prompt_sha", "feature"):
        value = getattr(tags, name, None)
        if isinstance(value, str) and value:
            attributes[f"langsmith.metadata.{name}"] = value
    span = _tracer.start_span("llm.call", attributes=attributes)
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield span
    finally:
        otel_context.detach(token)
