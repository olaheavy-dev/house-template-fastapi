"""Agents and the gateway through the redacting chat model (#78): `gateway.model()` is wrapped when
redaction is on, an agent's tool gets the original arguments, the Postgres checkpoint holds the
originals, and every call's llm_call row counts its redactions. Real PostgreSQL where it matters."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.core.budget import LlmCall
from app.llm.core.redaction import redaction_scope
from app.llm.core.steps import StepLimit
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel
from app.llm.langchain.steps import StepLimitMiddleware, step_limit_reached
from app.settings import Settings
from tests import wire
from tests.wire import EMAIL, PROVIDERS, TRICKY, Contact

pytestmark = pytest.mark.anyio

FINAL = {
    "name": "[NAME_1]",
    "email": "[EMAIL_1]",
    "address": {"city": "Leeds", "line": "1 High St"},
}
EXPECTED = {"name": TRICKY, "email": EMAIL, "address": {"city": "Leeds", "line": "1 High St"}}


# --- the gateway hands out the wrapper only when redaction is on ---------------------------------


def make_gateway(settings: Settings, **changes: Any) -> Gateway:
    return Gateway(settings.model_copy(update=changes), lambda: None)  # type: ignore[arg-type,return-value]


def test_the_gateways_model_is_wrapped_when_redaction_is_on_and_is_the_same_object_each_time(
    settings: Settings,
) -> None:
    gateway = make_gateway(settings)
    model = gateway.model()
    assert isinstance(model, RedactingChatModel)
    assert gateway.model() is model
    # The budget sits on the inner model, so it sees redacted text.
    assert isinstance(model.inner.callbacks, list)
    assert gateway.guard in model.inner.callbacks


def test_the_gateways_model_is_the_provider_model_when_redaction_is_off(
    settings: Settings,
) -> None:
    gateway = make_gateway(settings, llm_redaction="off")
    assert not isinstance(gateway.model(), RedactingChatModel)
    assert gateway.model() is gateway._model


def test_the_gateway_wraps_whatever_model_is_set_after_it_was_built(settings: Settings) -> None:
    gateway = make_gateway(settings)
    first = gateway.model()
    fake = wire.raw_model("anthropic")
    gateway._model = fake
    second = gateway.model()
    assert second is not first
    assert isinstance(second, RedactingChatModel)
    assert second.inner is fake


async def test_with_redaction_off_the_text_goes_as_it_is_and_the_row_has_no_count(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway: Gateway = app.state.gateway
    gateway.redactor = None
    gateway._model = wire.raw_model("openai", callbacks=[gateway.guard])
    with wire.captured(lambda _b: wire.openai_text("ok")) as sent:
        await gateway.model().ainvoke([HumanMessage(f"Mail {EMAIL}")])
    assert EMAIL in sent.text
    assert list(await db_session.scalars(select(LlmCall.redacted_count))) == [None]


# --- an agent -----------------------------------------------------------------------------------


def tool_names(body: dict[str, Any]) -> list[str]:
    return [t.get("name") or t.get("function", {}).get("name") for t in body.get("tools", [])]


def agent_reply(provider: str) -> wire.Reply:
    """First call: the tool. After its result: the final answer, as a structured answer when the
    request offers the schema's tool (a ToolStrategy), else as JSON text."""

    def reply(body: dict[str, Any]) -> Any:
        sent = json.dumps(body)
        after_tool = (
            '"role": "tool"' in sent or "tool_result" in sent or "function_call_output" in sent
        )
        if (
            not after_tool
            and "find_person" in tool_names(body)
            and body.get("tool_choice") != {"type": "none"}
        ):
            return wire.tool_reply(
                provider, "find_person", {"who": "[NAME_1]", "mail": "[EMAIL_1]"}
            )
        if "Contact" in tool_names(body):
            return wire.tool_reply(provider, "Contact", FINAL)
        return wire.text_reply(provider, json.dumps(FINAL))

    return reply


def make_find_person(seen: list[dict[str, Any]]) -> Any:
    @tool
    def find_person(who: str, mail: str) -> str:
        """Look a person up."""
        seen.append({"who": who, "mail": mail})
        return f"{who} <{mail}> is a customer"

    return find_person


def as_dict(parsed: Any) -> dict[str, Any]:
    return parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_an_agent_gives_its_tool_the_originals_and_the_checkpoint_keeps_them(
    provider: str, app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway: Gateway = app.state.gateway
    gateway._model = wire.raw_model(provider, callbacks=[gateway.guard])
    seen: list[dict[str, Any]] = []
    agent = gateway.agent(
        [make_find_person(seen)], system_prompt="You look people up.", response_format=Contact
    )
    thread_id = f"test-{uuid.uuid4()}"
    config: Any = {"configurable": {"thread_id": thread_id}}
    try:
        async with redaction_scope(known={"NAME": [TRICKY]}):
            with wire.captured(agent_reply(provider)) as sent:
                state = await agent.ainvoke(
                    {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]}, config
                )
            assert len(sent.bodies) == 2
            assert TRICKY not in sent.text
            assert EMAIL not in sent.text
            assert seen == [{"who": TRICKY, "mail": EMAIL}]  # the tool gets the originals
            assert as_dict(state["structured_response"]) == EXPECTED

            # Test 20: the saved thread holds the originals only, read back from Postgres.
            saved = await agent.aget_state(config)
            dump = repr(saved.values["messages"])
            assert EMAIL in dump
            assert not re.search(r"\[(NAME|EMAIL)_\d+\]", dump)
            stored = await db_session.execute(
                text(
                    "SELECT blob FROM checkpoint_blobs WHERE thread_id = :t "
                    "UNION ALL SELECT blob FROM checkpoint_writes WHERE thread_id = :t"
                ),
                {"t": thread_id},
            )
            blobs = b"".join(bytes(row[0]) for row in stored if row[0] is not None)
            assert EMAIL.encode() in blobs  # the original is what is stored...
            assert not re.search(rb"\[(NAME|EMAIL)_\d+\]", blobs)  # ...never a placeholder

            # The saved thread is redacted again on the wire when it is replayed.
            with wire.captured(agent_reply(provider)) as again:
                await agent.ainvoke({"messages": [HumanMessage("And again?")]}, config)
            assert TRICKY not in again.text
            assert EMAIL not in again.text
    finally:
        assert gateway.checkpointer is not None
        await gateway.checkpointer.adelete_thread(thread_id)

    # Test 23: each model call has its own row with its own count. The first run made two calls
    # (a tool call, then the answer) and the replay made one, which already had the tool result.
    rows = await db_session.scalars(select(LlmCall.redacted_count).order_by(LlmCall.redacted_count))
    counts = [count for count in rows if count is not None]
    assert len(counts) == 3  # no row without a count
    assert min(counts) >= 2
    if provider == "openai":
        # Call 1: the human text holds the name and the email: 2. Call 2 sends that again, plus
        # the tool call's arguments (name, email) and the tool's result (name, email): 6.
        assert counts[:2] == [2, 6]
        assert counts[2] > counts[1]  # the replay carries the whole thread, answer included


# --- the same requests as the raw model, whatever the strategy -----------------------------------

STRATEGIES: dict[str, Any] = {
    "auto": Contact,
    "provider": ProviderStrategy(Contact),
    "tool": ToolStrategy(Contact),
}


def build_agent(
    model: Any, *, limit: int, response_format: Any, seen: list[dict[str, Any]] | None = None
) -> Any:
    seen = [] if seen is None else seen
    return create_agent(
        model,
        [make_find_person(seen)],
        system_prompt="You look people up.",
        middleware=[StepLimitMiddleware(StepLimit(limit))],
        checkpointer=InMemorySaver(),
        response_format=response_format,
    )


@pytest.mark.parametrize("strategy", list(STRATEGIES))
@pytest.mark.parametrize("limit", [1, 13])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_every_strategy_and_the_step_limit_send_the_raw_models_requests_when_nothing_matches(
    provider: str, limit: int, strategy: str
) -> None:
    # create_agent probes the model (profile, name, type) to pick provider-native or tool-based
    # structured output, and the step limit's last call is made without tools (or with
    # tool_choice "none" on Anthropic). The wrapper must not change any of that.
    inner, model = wire.wrapped(provider)
    messages = {"messages": [HumanMessage("Find the Leeds flat")]}
    config: Any = {"configurable": {"thread_id": "a"}}
    with wire.captured(agent_reply(provider)) as raw:
        first = await build_agent(inner, limit=limit, response_format=STRATEGIES[strategy]).ainvoke(
            messages, config
        )
    with wire.captured(agent_reply(provider)) as wrapped:
        second = await build_agent(
            model, limit=limit, response_format=STRATEGIES[strategy]
        ).ainvoke(messages, config)
    assert wrapped.bodies == raw.bodies
    assert step_limit_reached(first) == step_limit_reached(second) == (limit == 1)
    if first.get("structured_response") is not None:
        assert as_dict(first["structured_response"]) == as_dict(second["structured_response"])


# --- streaming surfaces: nothing from the inner run, nothing twice -------------------------------

FINAL_PIECES = ["Hello [NA", "ME_1], mail ", "[EMAIL", "_1] ok"]
FINAL_TEXT = f"Hello {TRICKY}, mail {EMAIL} ok"
TOOL_PIECES = ['{"who": "[NA', 'ME_1]", "mail": "[EMAIL', '_1]"}']
STREAMED = ["openai", "openai-responses", "anthropic"]


def stream_reply(provider: str) -> wire.Reply:
    def reply(body: dict[str, Any]) -> Any:
        sent = json.dumps(body)
        markers = ('"role": "tool"', "tool_result", "function_call_output")
        after_tool = any(marker in sent for marker in markers)
        if provider == "openai-responses":
            if after_tool:
                return wire.responses_stream_text(FINAL_PIECES)
            return wire.responses_stream_tool("find_person", TOOL_PIECES)
        if provider == "openai":
            if after_tool:
                return wire.openai_stream_text(FINAL_PIECES)
            return wire.openai_stream_tool("find_person", TOOL_PIECES)
        if after_tool:
            return wire.anthropic_stream_text(FINAL_PIECES)
        deltas = [{"type": "input_json_delta", "partial_json": p} for p in TOOL_PIECES]
        start = {"type": "tool_use", "id": "toolu_1", "name": "find_person", "input": {}}
        return wire.anthropic_stream([(start, deltas)])

    return reply


def no_placeholder(text_: str) -> None:
    assert not re.search(r"\[(NAME|EMAIL)_\d+\]", text_), text_


@pytest.mark.parametrize("provider", STREAMED)
async def test_agent_astream_in_messages_mode_shows_restored_text_once(provider: str) -> None:
    _, model = wire.wrapped(provider)
    agent = build_agent(model, limit=13, response_format=None)
    texts: list[str] = []
    arguments: list[str] = []
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(stream_reply(provider)) as sent:
            async for chunk, meta in agent.astream(
                {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]},
                {"configurable": {"thread_id": "s"}},
                stream_mode="messages",
            ):
                if meta.get("langgraph_node") != "model":
                    continue
                texts.append(chunk.text)
                arguments += [c["args"] or "" for c in chunk.tool_call_chunks]
                no_placeholder(repr(chunk.content) + repr(chunk.tool_call_chunks))
    assert len(sent.bodies) == 2
    assert "".join(texts) == FINAL_TEXT  # once: the inner run's tokens are not streamed too
    assert json.loads("".join(arguments)) == {"who": TRICKY, "mail": EMAIL}
    # What the thread keeps (and would send again next turn) is the originals, never a placeholder.
    saved = await agent.aget_state({"configurable": {"thread_id": "s"}})
    stored = repr(saved.values["messages"])
    no_placeholder(stored)
    assert EMAIL in stored


@pytest.mark.parametrize("provider", STREAMED)
async def test_agent_astream_events_v2_announces_one_model_run_per_call(provider: str) -> None:
    _, model = wire.wrapped(provider)
    agent = build_agent(model, limit=13, response_format=None)
    texts: list[str] = []
    starts = 0
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(stream_reply(provider)):
            async for event in agent.astream_events(
                {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]},
                {"configurable": {"thread_id": "e"}},
                version="v2",
            ):
                if event["event"] == "on_chat_model_start":
                    starts += 1
                    no_placeholder(repr(event["data"]))
                if event["event"] == "on_chat_model_stream":
                    texts.append(event["data"]["chunk"].text)
                    no_placeholder(repr(event["data"]["chunk"].content))
                if event["event"] == "on_chat_model_end":
                    no_placeholder(repr(event["data"]["output"]))
    assert starts == 2  # the inner run is not announced
    assert "".join(texts) == FINAL_TEXT


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
@pytest.mark.parametrize("provider", STREAMED)
async def test_agent_stream_events_v3_shows_no_placeholder(provider: str) -> None:
    _, model = wire.wrapped(provider)
    agent = build_agent(model, limit=13, response_format=None)
    seen: list[str] = []
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(stream_reply(provider)):
            run = await agent.astream_events(
                {"messages": [HumanMessage(f"Find {TRICKY} at {EMAIL}")]},
                {"configurable": {"thread_id": "v3"}},
                version="v3",
            )
            async for event in run:
                seen.append(repr(event))
            final = await run.output()
    assert seen
    no_placeholder("\n".join(seen))
    assert final["messages"][-1].text == FINAL_TEXT
