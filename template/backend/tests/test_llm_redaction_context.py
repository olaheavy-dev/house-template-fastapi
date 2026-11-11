"""The caller's context reaches the hidden inner run on every path (#78): metadata (feature,
user_ref, prompt), tags and tracers. langchain-core's `astream` and its streaming branch of
`ainvoke` call `_astream` without a run manager, so a wrapper that read the context only from a
run manager would reserve budget with no user: the per-user cap would be skipped."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langsmith import Client
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import User
from app.llm.core.budget import LlmCall, UserBudgetExhausted
from app.llm.core.redaction import RedactionFailed, Redactor
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel, masking_anonymizer
from tests import wire
from tests.test_llm_redaction_agents import build_agent

pytestmark = pytest.mark.anyio

USER = User(id="u_42")


class StreamingHandler(wire.Recorder, _StreamingCallbackHandler[Any]):
    """What LangGraph's message stream and astream_events attach: a streaming handler, which makes
    ainvoke take its streaming branch."""

    def tap_output_aiter(self, run_id: Any, output: Any) -> Any:
        return output

    def tap_output_iter(self, run_id: Any, output: Any) -> Any:
        return output


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(await session.scalars(select(LlmCall)))


def gateway_with_wire(app: FastAPI) -> Gateway:
    gateway: Gateway = app.state.gateway
    gateway._model = wire.raw_model("openai", callbacks=[gateway.guard])
    return gateway


async def test_a_streamed_call_records_its_user_and_feature(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway = gateway_with_wire(app)
    config = gateway.call_config(USER, "answer")
    with wire.captured(lambda _b: wire.openai_stream_text(["a", "b"])):
        _ = [c async for c in gateway.model().astream([HumanMessage("hi")], config=config)]
    [row] = await rows(db_session)
    assert (row.user_ref, row.feature) == ("u_42", "answer")
    assert row.status == "settled"


async def test_an_ainvoke_that_streams_records_its_user_and_feature_too(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway = gateway_with_wire(app)
    config = gateway.call_config(USER, "answer")
    config["callbacks"] = [StreamingHandler()]
    with wire.captured(lambda _b: wire.openai_stream_text(["a", "b"])):
        await gateway.model().ainvoke([HumanMessage("hi")], config=config)
    [row] = await rows(db_session)
    assert (row.user_ref, row.feature) == ("u_42", "answer")


async def test_a_streamed_agent_records_its_user_and_feature(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway = gateway_with_wire(app)
    agent = build_agent(gateway.model(), limit=13, response_format=None)
    config = gateway.call_config(USER, "agent")
    config["configurable"] = {"thread_id": "ctx"}
    with wire.captured(lambda _b: wire.openai_stream_text(["a", "b"])):
        async for _ in agent.astream(
            {"messages": [HumanMessage("hi")]}, config, stream_mode="messages"
        ):
            pass
    [row] = await rows(db_session)
    assert (row.user_ref, row.feature) == ("u_42", "agent")


async def test_the_users_daily_cap_refuses_a_streamed_call_before_anything_is_sent(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway = gateway_with_wire(app)
    gateway.guard.user_cap = 1  # micro-USD: no call fits
    config = gateway.call_config(USER, "answer")
    with wire.captured(lambda _b: wire.openai_stream_text(["a"])) as sent:
        with pytest.raises(UserBudgetExhausted) as refused:
            _ = [c async for c in gateway.model().astream([HumanMessage("hi")], config=config)]
        with pytest.raises(UserBudgetExhausted):
            await gateway.model().ainvoke(
                [HumanMessage("hi")], config={**config, "callbacks": [StreamingHandler()]}
            )
    assert refused.value.status == 429
    assert sent.bodies == []


# --- traces and tags -----------------------------------------------------------------------------


def runs_sent(session: wire.FakeLangSmithSession) -> list[dict[str, Any]]:
    found = []
    for line in session.sent:
        method, url, data = line.split(" ", 2)
        if method == "POST" and url.endswith("/runs") and data.strip().startswith("{"):
            found.append(json.loads(data))
    return found


@pytest.mark.parametrize("path", ["astream", "ainvoke-streaming"])
async def test_the_inner_run_is_a_child_of_the_wrapper_run_when_streaming(path: str) -> None:
    session = wire.FakeLangSmithSession()
    client = Client(
        api_url="http://langsmith.invalid",
        api_key="k",
        session=session,
        auto_batch_tracing=False,
        anonymizer=masking_anonymizer(Redactor(wire.DETECTORS)),
    )
    tracer = LangChainTracer(client=client, project_name="test")
    recorder = wire.Recorder()
    inner = wire.raw_model("openai", callbacks=[recorder])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    config: Any = {"callbacks": [tracer], "tags": ["mine"], "metadata": {"feature": "answer"}}
    with wire.captured(lambda _b: wire.openai_stream_text(["a", "b"])):
        if path == "astream":
            _ = [c async for c in model.astream([HumanMessage("hi")], config=config)]
        else:
            config["callbacks"] = [tracer, StreamingHandler()]
            await model.ainvoke([HumanMessage("hi")], config=config)
    wait_for_all_tracers()
    runs = runs_sent(session)
    by_id = {run["id"]: run for run in runs}
    children = [run for run in runs if run.get("parent_run_id") in by_id]
    assert len(runs) == 2
    assert len(children) == 1  # the inner run hangs under the wrapper's
    # The inner run carries the caller's metadata and tags, and `nostream`.
    [(_, _, metadata)] = recorder.starts
    assert metadata["feature"] == "answer"
    assert metadata["redacted_count"] == 0


async def test_the_inner_run_carries_the_callers_tags_and_nostream() -> None:
    tags_seen: list[list[str]] = []

    class Tags(wire.Recorder):
        async def on_chat_model_start(self, *args: Any, tags: Any = None, **kwargs: Any) -> None:
            tags_seen.append(list(tags or []))

    inner = wire.raw_model("openai", callbacks=[Tags()])
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    with wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage("hi")], config={"tags": ["mine"]})
    with wire.captured(lambda _b: wire.openai_stream_text(["ok"])):
        _ = [c async for c in model.astream([HumanMessage("hi")], config={"tags": ["mine"]})]
    assert len(tags_seen) == 2
    for tags in tags_seen:
        assert "mine" in tags
        assert "nostream" in tags


async def test_a_streamed_call_with_no_context_to_carry_is_refused_not_sent_without_it() -> None:
    # Fail closed: _astream called straight, with neither a run manager nor a captured context.
    inner = wire.raw_model("openai")
    model = RedactingChatModel(inner, Redactor(wire.DETECTORS))
    with (
        wire.captured(lambda _b: wire.openai_stream_text(["a"])) as sent,
        pytest.raises(RedactionFailed),
    ):
        _ = [c async for c in model._astream([HumanMessage("hi")])]
    assert sent.bodies == []
