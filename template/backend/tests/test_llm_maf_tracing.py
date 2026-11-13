"""Tracing MAF model calls (#35 PR 7, #78 part 4): what reaches LangSmith, and what never does.
Spans go to an in-memory exporter behind the same filter the LangSmith exporter sits behind, so
nothing leaves the process. The values are synthetic."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest
from agent_framework import ChatResponse, Content, Message, tool
from agent_framework.observability import OBSERVABILITY_SETTINGS
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.auth import User
from app.llm.core.budget import LlmCall
from app.llm.core.errors import ProviderError
from app.llm.core.prompts import Prompt
from app.llm.maf import tracing
from app.llm.maf.gateway import Gateway
from app.settings import Settings
from tests.harness import Failure, Reply
from tests.harness.maf import HARNESS

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 14)
PHRASE = "periwinkle-umbrella-4417"  # a phrase that must appear in no span and no log
PROMPT = Prompt(name="summary", text="Summarise the user's message in one line.")


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def spans(settings: Settings, sessions: async_sessionmaker[AsyncSession]) -> Any:
    """A gateway whose spans are exported to memory, and a reader of what was exported."""

    def build(**options: Any) -> tuple[Gateway, InMemorySpanExporter]:
        configured = settings.model_copy(
            update={"llm_model": "anthropic:claude-sonnet-5-5", **options}
        )
        gateway = Gateway(configured, lambda: sessions)
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        memory = InMemorySpanExporter()
        gateway.tracing = tracing.configure(configured, exporter=memory)
        return gateway, memory

    return build


def exported(gateway: Gateway, memory: InMemorySpanExporter) -> list[ReadableSpan]:
    gateway.tracing.shutdown()
    return list(memory.get_finished_spans())


def text_of(spans: Sequence[ReadableSpan]) -> str:
    parts = []
    for span in spans:
        parts.append(span.name)
        parts.append(repr(dict(span.attributes or {})))
        for event in span.events:
            parts.append(event.name + repr(dict(event.attributes or {})))
    return "\n".join(parts)


async def test_sensitive_data_on_with_redaction_on_refuses_to_start(settings: Settings) -> None:
    with pytest.raises(ValidationError, match="ENABLE_SENSITIVE_DATA is on while LLM_REDACTION"):
        Settings(
            _env_file=None,
            app_env="test",
            enable_sensitive_data=True,
            llm_redaction="on",
        )
    allowed = Settings(
        _env_file=None, app_env="test", enable_sensitive_data=True, llm_redaction="off"
    )
    assert allowed.enable_sensitive_data


async def test_the_call_span_and_the_ledger_row_carry_the_same_prompt_hash(
    spans: Any, db_session: AsyncSession
) -> None:
    gateway, memory = spans()
    HARNESS.install_scripted(gateway, [Reply("one line"), Reply("another line")])
    changed = Prompt(name="summary", text=PROMPT.text + " Be brief.")
    for prompt in (PROMPT, changed):
        with gateway.call_options(User(id="u_1"), "summary", prompt):
            await gateway.client().get_response([Message("user", ["hello"])])
    calls = [span for span in exported(gateway, memory) if span.name == "llm.call"]
    shas = [dict(span.attributes or {})["langsmith.metadata.prompt_sha"] for span in calls]
    assert shas == [PROMPT.sha, changed.sha]
    assert PROMPT.sha != changed.sha  # changing the prompt file changes the hash
    rows = {
        row.prompt_sha: row
        for row in await db_session.scalars(select(LlmCall).where(LlmCall.day == DAY))
    }
    for span in calls:
        attributes = dict(span.attributes or {})
        row = rows[str(attributes["langsmith.metadata.prompt_sha"])]
        assert attributes["langsmith.metadata.llm_call_id"] == str(row.id)
        assert attributes["langsmith.metadata.prompt"] == "summary"
        assert attributes["langsmith.metadata.feature"] == "summary"


async def test_only_mafs_spans_and_the_call_span_are_sent_and_the_chat_span_is_its_child(
    spans: Any,
) -> None:
    gateway, memory = spans()
    HARNESS.install_scripted(gateway, [Reply("ok")])
    with trace.get_tracer("fastapi").start_as_current_span("GET /summary"):
        await gateway.client().get_response([Message("user", ["hello"])])
    sent = exported(gateway, memory)
    scopes = {span.instrumentation_scope.name for span in sent if span.instrumentation_scope}
    assert scopes <= {"agent_framework", "app.llm"}
    assert "GET /summary" not in [span.name for span in sent]
    [call] = [span for span in sent if span.name == "llm.call"]
    chats = [span for span in sent if span.name.startswith("chat")]
    assert chats
    assert all(
        span.parent is not None and span.parent.span_id == call.context.span_id for span in chats
    )


async def test_no_text_reaches_a_span_or_a_log(spans: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # The phrase is in the question, the answer, a tool's arguments and its result. MAF's own
    # switch is also set in the environment: the gateway sets capture explicitly, so it cannot
    # turn it on.
    monkeypatch.setenv("ENABLE_SENSITIVE_DATA", "true")
    gateway, memory = spans()

    @tool
    def lookup(term: str) -> str:
        """Look a term up."""
        return f"found {term}"

    asks = ChatResponse(
        messages=[
            Message(
                "assistant",
                [Content.from_function_call("c1", "lookup", arguments={"term": PHRASE})],
            )
        ]
    )
    HARNESS.install_scripted(gateway, [asks, Reply(f"The answer is {PHRASE}.")])
    with capture_logs() as logs, gateway.call_options(User(id="u_1"), "summary", PROMPT):
        await gateway.agent([lookup], name="t", instructions=f"Know {PHRASE}.").run(
            f"What is {PHRASE}?"
        )
    sent = exported(gateway, memory)
    assert sent
    assert PHRASE not in text_of(sent)
    assert PHRASE not in repr(logs)


async def test_the_check_can_see_text_when_sensitive_data_is_on(spans: Any) -> None:
    # The negative control for the test above: with capture on (allowed only with redaction off),
    # the same reading finds the phrase in MAF's spans.
    gateway, memory = spans(enable_sensitive_data=True, llm_redaction="off")
    HARNESS.install_scripted(gateway, [Reply(f"The answer is {PHRASE}.")])
    await gateway.client().get_response([Message("user", [f"What is {PHRASE}?"])])
    assert PHRASE in text_of(exported(gateway, memory))


async def test_a_failed_call_still_sends_its_span_with_the_error_kind(spans: Any) -> None:
    # The calls that most need a trace are the ones that failed (PR #122 review): the span is
    # ended and sent with an error status and the failure's kind, never its message.
    gateway, memory = spans()
    HARNESS.install_scripted(gateway, [Failure("rate_limit", message=f"echo {PHRASE}")])
    with gateway.call_options(User(id="u_1"), "summary", PROMPT), pytest.raises(ProviderError):
        await gateway.client().get_response([Message("user", ["hello"])])
    sent = exported(gateway, memory)
    [call] = [span for span in sent if span.name == "llm.call"]
    assert call.status.status_code is StatusCode.ERROR
    attributes = dict(call.attributes or {})
    assert attributes["error.type"] == "rate_limited"
    assert attributes["langsmith.metadata.prompt_sha"] == PROMPT.sha
    assert PHRASE not in text_of(sent)


async def test_a_sampling_rate_of_zero_sends_nothing(spans: Any) -> None:
    gateway, memory = spans(langsmith_tracing_sampling_rate=0.0)
    HARNESS.install_scripted(gateway, [Reply("ok")])
    await gateway.client().get_response([Message("user", ["hello"])])
    assert exported(gateway, memory) == []


class _Stuck(SpanExporter):
    """An exporter whose endpoint never answers."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        time.sleep(30)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


async def test_shutdown_is_bounded_when_the_endpoint_hangs(settings: Settings) -> None:
    configured = tracing.configure(settings, exporter=_Stuck())
    with trace.get_tracer(tracing.SCOPE).start_as_current_span("llm.call"):
        pass
    started = time.monotonic()
    configured.shutdown(timeout_seconds=0.5)
    assert time.monotonic() - started < 2


def test_tracing_is_off_in_tests_whatever_the_settings(settings: Settings) -> None:
    # Tracing asked for, with a key: still nothing is sent from a test run.
    asked = settings.model_copy(
        update={"langsmith_tracing": True, "langsmith_api_key": SecretStr("lsv2-not-a-key")}
    )
    configured = tracing.configure(asked, exporter=None)
    assert configured.processor is None


def test_configuring_keeps_maf_from_installing_providers_of_its_own(settings: Settings) -> None:
    # MAF's configure_otel_providers replaces the global providers (a second exporter, and
    # sensitive capture as its arguments say); once the gateway has configured tracing it is a
    # no-op.
    OBSERVABILITY_SETTINGS._executed_setup = False
    tracing.configure(settings)
    assert OBSERVABILITY_SETTINGS._executed_setup is True
    assert OBSERVABILITY_SETTINGS.enable_instrumentation is True


def test_the_private_maf_setting_the_gateway_sets_exists_in_the_pinned_core() -> None:
    # tracing.configure marks MAF's own provider setup as done (agent-framework-core 1.25.0, a
    # private attribute); a version that renames it must fail here, not install a second exporter.
    assert hasattr(OBSERVABILITY_SETTINGS, "_executed_setup")
