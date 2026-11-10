"""LangSmith traces are masked (#78, test 25): the client the gateway builds carries an anonymizer
made from the same redactor, so what a trace holds is placeholders, whichever run it is (the
wrapper's public run, the hidden inner run, a graph's run, a tool's) and however tracing started.
The LangSmith client is real; its HTTP session is replaced, so nothing leaves the machine."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer, get_client, wait_for_all_tracers
from langsmith import Client, configure, tracing_context
from langsmith.schemas import LangSmithInfo
from pydantic import SecretStr
from structlog.testing import capture_logs

from app.llm.core.redaction import Redactor, current_scope, redaction_scope
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel, masking_anonymizer
from app.settings import Settings
from tests import wire
from tests.test_llm_redaction_agents import build_agent
from tests.wire import DETECTORS, EMAIL, NAME, PHONE, TRICKY

pytestmark = pytest.mark.anyio

INFO = LangSmithInfo()


@pytest.fixture(autouse=True)
def clear_the_process_wide_langsmith_client() -> Iterator[None]:
    yield
    configure(client=None)  # the gateway sets it; leave no test client behind


def mask_client(
    redactor: Redactor | None = None, **options: Any
) -> tuple[Client, wire.FakeLangSmithSession]:
    session = wire.FakeLangSmithSession()
    options.setdefault("auto_batch_tracing", False)  # each run is posted as it ends
    client = Client(
        api_url="http://langsmith.invalid",
        api_key="k",
        session=session,
        anonymizer=masking_anonymizer(redactor or Redactor(DETECTORS)),
        **options,
    )
    return client, session


def plain_agent_reply(provider: str) -> wire.Reply:
    def reply(body: dict[str, Any]) -> Any:
        sent = json.dumps(body)
        if '"role": "tool"' in sent or "tool_result" in sent:
            return wire.text_reply(provider, "Hi [NAME_1]")
        return wire.tool_reply(provider, "find_person", {"who": "[NAME_1]", "mail": "[EMAIL_1]"})

    return reply


def assert_masked(session: wire.FakeLangSmithSession) -> None:
    for seed in (TRICKY, EMAIL, NAME, PHONE):
        assert seed not in session.text
    assert json.dumps(TRICKY)[1:-1] not in session.text  # nor JSON-escaped


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_an_explicit_tracer_records_the_public_and_the_hidden_run_masked(
    provider: str,
) -> None:
    _, model = wire.wrapped(provider)
    client, session = mask_client()
    tracer = LangChainTracer(client=client, project_name="test")
    # A name no pattern finds: only the request's scope names it.
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.text_reply(provider, "Hi [NAME_1] at [EMAIL_1]")):
            answer = await model.ainvoke(
                [HumanMessage(f"{TRICKY} {EMAIL}")], config={"callbacks": [tracer]}
            )
        wait_for_all_tracers()
    assert TRICKY in answer.text  # the caller still gets the originals
    assert session.text.count("POST") >= 2  # the wrapper's run and its inner child run
    assert_masked(session)
    assert "[EMAIL_1]" in session.text


async def test_without_the_anonymizer_the_public_run_records_the_originals() -> None:
    # The negative control: it is the anonymizer that masks, not something else in the path. The
    # hidden inner run already holds placeholders only; the caller's own run holds the originals.
    session = wire.FakeLangSmithSession()
    client = Client(
        api_url="http://langsmith.invalid", api_key="k", session=session, auto_batch_tracing=False
    )
    _, model = wire.wrapped("openai")
    tracer = LangChainTracer(client=client, project_name="test")
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_text("ok")):
            await model.ainvoke([HumanMessage(f"{TRICKY} {EMAIL}")], config={"callbacks": [tracer]})
        wait_for_all_tracers()
    assert EMAIL in session.text
    assert "[EMAIL_1]" in session.text


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_a_graph_run_started_by_tracing_context_has_every_run_masked(provider: str) -> None:
    _, model = wire.wrapped(provider)
    client, session = mask_client()
    seen: list[dict[str, Any]] = []
    agent = build_agent(model, limit=13, response_format=None, seen=seen)
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with (
            tracing_context(enabled=True, client=client, project_name="test"),
            wire.captured(plain_agent_reply(provider)),
        ):
            await agent.ainvoke(
                {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]},
                {"configurable": {"thread_id": "tr"}},
            )
        wait_for_all_tracers()
        client.flush()
    assert seen == [{"who": TRICKY, "mail": EMAIL}]  # the tool did get the originals
    assert session.text.count("POST") >= 4  # the graph, model steps, hidden runs and the tool
    assert_masked(session)


async def test_tracing_started_from_the_environment_uses_the_gateways_masking_client(
    settings: Settings,
) -> None:
    # The gateway sets the process-wide client when redaction is on, so a tracer (or the
    # environment's own tracing) that names no client of its own is masked too.
    gateway = Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]
    session = wire.FakeLangSmithSession()
    client = get_client()
    client.session = session
    assert isinstance(gateway.model(), RedactingChatModel)
    tracer = LangChainTracer(project_name="test")
    assert tracer.client is client
    _, model = wire.wrapped("openai")
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_text("Hi [NAME_1]")):
            await model.ainvoke([HumanMessage(f"{TRICKY} {EMAIL}")], config={"callbacks": [tracer]})
        wait_for_all_tracers()
        client.flush()
    assert_masked(session)


async def test_tracing_enabled_with_no_client_of_its_own_runs_through_the_gateways_client(
    settings: Settings,
) -> None:
    # What LANGSMITH_TRACING=true does, without the gateway's own tracer: LangChain builds a
    # tracer for the run, and it takes the process-wide client.
    Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]
    session = wire.FakeLangSmithSession()
    client = get_client()
    client.session = session
    _, model = wire.wrapped("openai")
    seen: list[dict[str, Any]] = []
    agent = build_agent(model, limit=13, response_format=None, seen=seen)
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with (
            tracing_context(enabled=True, project_name="test"),
            wire.captured(plain_agent_reply("openai")),
        ):
            await agent.ainvoke(
                {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]},
                {"configurable": {"thread_id": "env"}},
            )
        wait_for_all_tracers()
        client.flush()
    assert session.text.count("POST") >= 4
    assert_masked(session)


async def test_the_scope_is_known_to_the_anonymizer_when_runs_are_batched() -> None:
    # Runs are sent from the tracer's own worker thread, which runs in a copy of the request's
    # context, so the anonymizer sees this request's known values.
    base = masking_anonymizer(Redactor(DETECTORS))
    scopes: list[bool] = []
    outputs: list[str] = []

    def anonymizer(data: dict[str, Any]) -> dict[str, Any]:
        scopes.append(bool(current_scope().known))
        masked = base(data)
        outputs.append(json.dumps(masked, default=str))
        return masked

    client = Client(
        api_url="http://langsmith.invalid",
        api_key="k",
        session=wire.FakeLangSmithSession(),
        anonymizer=anonymizer,  # batching stays on
    )
    _, model = wire.wrapped("openai")
    tracer = LangChainTracer(client=client, project_name="test")
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_text("Hi [NAME_1]")):
            await model.ainvoke([HumanMessage(f"{TRICKY} {EMAIL}")], config={"callbacks": [tracer]})
        wait_for_all_tracers()
    assert scopes
    assert all(scopes)
    joined = "".join(outputs)
    assert TRICKY not in joined
    assert EMAIL not in joined
    client.flush()


async def test_an_anonymizer_that_fails_sends_nothing_and_logs_only_the_error_type() -> None:
    class Boom:
        kind = "EMAIL"

        def find(self, text: str) -> Any:
            raise RuntimeError(f"cannot read {text}")  # an error message that holds the text

    client, session = mask_client(Redactor([Boom()]))
    _, model = wire.wrapped("openai")  # the model's own redactor works; only tracing's is broken
    tracer = LangChainTracer(client=client, project_name="test")
    with capture_logs() as logs, wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage(EMAIL)], config={"callbacks": [tracer]})
        wait_for_all_tracers()
    assert EMAIL not in session.text
    events = [e for e in logs if e["event"] == "llm_redaction_failed"]
    assert events
    assert all(e["error_type"] == "RuntimeError" for e in events)
    assert EMAIL not in repr(logs)


async def test_a_run_that_errors_while_the_anonymizer_fails_sends_no_value_at_all() -> None:
    # LangSmith keeps a run's original error unless the anonymizer's result still has an "error"
    # key, so a bare {} on failure would let the error (and its email) through.
    class Boom:
        kind = "EMAIL"

        def find(self, text: str) -> Any:
            raise RuntimeError("masking is broken")

    class Raising(GenericFakeChatModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError(f"no account for {EMAIL}")

    client, session = mask_client(Redactor([Boom()]))
    tracer = LangChainTracer(client=client, project_name="test")
    model = RedactingChatModel(Raising(messages=iter([])), Redactor(DETECTORS))
    with pytest.raises(ValueError, match="no account"):
        await model.ainvoke([HumanMessage(f"Reach {EMAIL}")], config={"callbacks": [tracer]})
    wait_for_all_tracers()
    assert session.text.count("PATCH") + session.text.count("POST") >= 2
    assert EMAIL not in session.text
    assert "no account for" not in session.text
    assert "[redaction failed]" in session.text


def test_the_anonymizer_keeps_the_keys_with_a_neutral_value_for_a_shape_it_cannot_mask() -> None:
    anonymize = masking_anonymizer(Redactor(DETECTORS))
    with (
        mock.patch("app.llm.langchain.redaction._walk_data", return_value=["not", "a", "dict"]),
        capture_logs() as logs,
    ):
        assert anonymize({"inputs": EMAIL, "error": EMAIL}) == {
            "inputs": "[redaction failed]",
            "error": "[redaction failed]",
        }
    assert EMAIL not in repr(logs)


async def test_hide_inputs_still_wins_over_masking() -> None:
    client, session = mask_client(hide_inputs=True, hide_outputs=True)
    _, model = wire.wrapped("openai")
    tracer = LangChainTracer(client=client, project_name="test")
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(lambda _b: wire.openai_text("Hi [NAME_1]")):
            await model.ainvoke([HumanMessage(f"{TRICKY} {EMAIL}")], config={"callbacks": [tracer]})
        wait_for_all_tracers()
    assert_masked(session)
    assert "[EMAIL_1]" not in session.text  # hidden entirely, not even as a placeholder
    assert "Hi " not in session.text


# --- the gateway builds and shares the client ----------------------------------------------------


def make_gateway(settings: Settings, **changes: Any) -> Gateway:
    return Gateway(settings.model_copy(update=changes), lambda: None)  # type: ignore[arg-type,return-value]


def test_with_redaction_on_the_gateway_sets_a_masking_client_for_the_process(
    settings: Settings,
) -> None:
    make_gateway(settings)
    client = get_client()
    assert client._anonymizer is not None
    # The tracing is off here (the test environment), so there is no thread for the client.
    assert client.tracing_queue is None


def test_the_gateways_tracer_uses_that_client_and_keeps_the_hide_settings(
    settings: Settings,
) -> None:
    with mock.patch.object(Client, "info", new_callable=mock.PropertyMock, return_value=INFO):
        gateway = _tracing_gateway(settings)
    assert gateway.tracer is not None
    assert gateway.tracer.client is get_client()
    client = gateway.tracer.client
    assert client._anonymizer is not None
    assert client._hide_outputs is True
    assert client._hide_inputs is False
    assert client.tracing_queue is not None  # batching, as a tracing client always had
    client.close(timeout=0)  # stop its background thread


def _tracing_gateway(settings: Settings) -> Gateway:
    return make_gateway(
        settings,
        app_env="production",
        langsmith_tracing=True,
        langsmith_project="demo",
        langsmith_api_key=SecretStr("lsv2-x"),
        langsmith_endpoint="http://langsmith.invalid",
        langsmith_hide_outputs=True,
        database_url=SecretStr("postgresql+psycopg://app:secret@db.internal:5432/app"),
    )


def test_with_redaction_off_the_gateway_sets_no_client_and_traces_as_before(
    settings: Settings,
) -> None:
    configure(client=None)
    gateway = make_gateway(settings, llm_redaction="off")
    from langsmith import run_trees

    assert run_trees._CLIENT is None
    assert gateway.tracer is None
    with mock.patch.object(Client, "info", new_callable=mock.PropertyMock, return_value=INFO):
        traced = make_gateway(
            settings,
            llm_redaction="off",
            app_env="production",
            langsmith_tracing=True,
            langsmith_api_key=SecretStr("lsv2-x"),
            langsmith_endpoint="http://langsmith.invalid",
            database_url=SecretStr("postgresql+psycopg://app:secret@db.internal:5432/app"),
        )
    assert traced.tracer is not None
    assert traced.tracer.client._anonymizer is None
    assert run_trees._CLIENT is None
    traced.tracer.client.close(timeout=0)


async def test_closing_the_gateway_flushes_the_process_wide_client(settings: Settings) -> None:
    gateway = make_gateway(settings)
    with mock.patch.object(Client, "flush") as flush:
        await gateway.aclose()
    flush.assert_called_once()
    assert get_client() is not None
