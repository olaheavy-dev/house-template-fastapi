"""Agent threads saved in Postgres: what the checkpointer can read back (#36)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import ormsgpack
import pytest
from fastapi import FastAPI
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import EXT_CONSTRUCTOR_SINGLE_ARG, JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, SecretStr, model_validator
from typing_extensions import TypedDict

from app.llm import checkpoint_types
from app.llm.langchain import checkpoints
from app.llm.langchain.gateway import Gateway
from app.settings import Settings

pytestmark = pytest.mark.anyio

USAGE: UsageMetadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


class ToolCallingFake(GenericFakeChatModel):
    """Answers from a script, and accepts tools as create_agent binds them."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


CALLS: list[str] = []
BUILT: list[int] = []
# LangGraph warns once per class and process, so each class below is used in exactly one test.
BLOCKED = "langgraph.checkpoint.serde.jsonplus"


def record_call(argument: str) -> str:
    """A stand-in for any function a planted row could name."""
    CALLS.append(argument)
    return "called"


class Listed(BaseModel):
    x: int


class Unlisted(BaseModel):
    x: int


class Canary(BaseModel):
    """Records each time it is validated, that is, built by the normal constructor."""

    x: int

    @model_validator(mode="after")
    def record_build(self) -> Canary:
        BUILT.append(self.x)
        return self


class CanaryState(TypedDict):
    canary: Canary


def blocked(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [m for m in (r.getMessage() for r in caplog.records) if m.startswith("Blocked")]


async def test_a_thread_saved_with_the_default_loader_loads_and_resumes(
    app: FastAPI, db_client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    # Versions 0.3.0 to 0.7.0 saved threads with LangGraph's default serializer. Whatever the
    # checkpointer loads with now must still read and resume them: nothing in a normal thread
    # (messages, a pending Send, usage metadata) may come back as plain data.
    gateway: Gateway = app.state.gateway
    strict = gateway.checkpointer
    assert isinstance(strict, AsyncPostgresSaver)
    script = [
        AIMessage(
            "",
            tool_calls=[{"name": "add", "args": {"a": 2, "b": 2}, "id": "call-1"}],
            usage_metadata=USAGE,
        ),
        AIMessage("2 + 2 is 4.", usage_metadata=USAGE),
    ]
    gateway._model = ToolCallingFake(messages=iter(script), callbacks=[gateway.guard])
    thread_id = f"test-{uuid.uuid4()}"
    thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    try:
        # Written the old way: a saver on the same pool with the default serializer. It stops
        # before the tools node, so the saved thread holds a pending Send.
        gateway.checkpointer = AsyncPostgresSaver(strict.conn)
        writer = gateway.agent([add], system_prompt="You add numbers.", interrupt_before=["tools"])
        await writer.ainvoke({"messages": [HumanMessage("What is 2 + 2?")]}, config=thread)

        # Read and resumed with whatever the gateway's checkpointer is now.
        gateway.checkpointer = strict
        reader = gateway.agent([add], system_prompt="You add numbers.", interrupt_before=["tools"])
        saved = await reader.aget_state(thread)
        assert saved.next == ("tools",)
        human, ai = saved.values["messages"]
        assert isinstance(human, HumanMessage)
        assert isinstance(ai, AIMessage)
        assert ai.tool_calls == [
            {"name": "add", "args": {"a": 2, "b": 2}, "id": "call-1", "type": "tool_call"}
        ]
        assert ai.usage_metadata == USAGE

        state = await reader.ainvoke(None, config=thread)
        assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "ai"]
        assert state["messages"][2].text == "4"  # add(2, 2)
        assert state["messages"][3].text == "2 + 2 is 4."
    finally:
        gateway.checkpointer = strict
        await strict.adelete_thread(thread_id)
    assert [r for r in caplog.records if "Blocked" in r.getMessage()] == []


@pytest.fixture
async def opened(settings: Settings, database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """The checkpointer as the lifespan opens it (the pool holds no connection until used)."""
    saver, pool = await checkpoints.open_checkpointer(
        settings.model_copy(update={"database_url": SecretStr(database_url)})
    )
    try:
        yield saver
    finally:
        await pool.close()


async def test_a_stored_row_cannot_make_the_checkpointer_call_a_function(
    opened: AsyncPostgresSaver, caplog: pytest.LogCaptureFixture
) -> None:
    # The default loader imports module.name and calls it with the argument, whatever the row
    # says. A row built by hand names a function in this module: strict loading hands back the
    # argument as data, and the function is never called.
    caplog.set_level(logging.WARNING, logger=BLOCKED)
    row = ormsgpack.packb(
        ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            ormsgpack.packb((__name__, "record_call", "planted")),
        )
    )
    CALLS.clear()
    loaded = opened.serde.loads_typed(("msgpack", row))
    assert loaded == "planted"
    assert CALLS == []
    assert blocked(caplog) == [
        f"Blocked deserialization of {__name__}.record_call - not in "
        "allowed_msgpack_modules. Add to allowed_msgpack_modules to allow: "
        f"[('{__name__}', 'record_call')]"
    ]

    # The same row under LangGraph's default loader does call it: the test would not catch
    # anything if the strict loader were no different.
    assert (
        JsonPlusSerializer(allowed_msgpack_modules=True).loads_typed(("msgpack", row)) == "called"
    )
    assert CALLS == ["planted"]
    CALLS.clear()


def test_a_listed_class_is_rebuilt_and_an_unlisted_one_comes_back_as_its_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=BLOCKED)
    strict = checkpoints.serializer([Listed])
    assert strict.loads_typed(strict.dumps_typed(Listed(x=1))) == Listed(x=1)
    assert strict.loads_typed(strict.dumps_typed(Unlisted(x=2))) == {"x": 2}
    # Once per class and process: loading it again logs nothing more.
    assert strict.loads_typed(strict.dumps_typed(Unlisted(x=3))) == {"x": 3}
    assert [m.split(" - ")[0] for m in blocked(caplog)] == [
        f"Blocked deserialization of {__name__}.Unlisted"
    ]
    # What LangGraph's own list holds still loads with nothing listed, and a list adds to it.
    message = HumanMessage("hi", id="m1")
    for serde in (checkpoints.serializer(), strict):
        assert serde.loads_typed(serde.dumps_typed(message)) == message


async def test_the_checkpointer_loads_what_the_project_lists(
    settings: Settings, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_types, "ALLOWED", (Listed,))
    saver, pool = await checkpoints.open_checkpointer(
        settings.model_copy(update={"database_url": SecretStr(database_url)})
    )
    try:
        assert saver.serde.loads_typed(saver.serde.dumps_typed(Listed(x=5))) == Listed(x=5)
    finally:
        await pool.close()


async def test_a_saved_object_of_an_unlisted_class_comes_back_as_its_fields_unbuilt(
    app: FastAPI, db_client: httpx.AsyncClient
) -> None:
    gateway: Gateway = app.state.gateway
    strict = gateway.checkpointer
    assert isinstance(strict, AsyncPostgresSaver)

    def node(state: CanaryState) -> CanaryState:
        return {"canary": Canary(x=7)}

    builder = StateGraph(CanaryState)
    builder.add_node("make", node)
    builder.add_edge(START, "make")
    builder.add_edge("make", END)
    thread_id = f"test-{uuid.uuid4()}"
    thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        # Saved the old way, with the default serializer, then read the new way.
        old = builder.compile(checkpointer=AsyncPostgresSaver(strict.conn))
        await old.ainvoke({"canary": Canary(x=1)}, config=thread)
        BUILT.clear()
        saved = await builder.compile(checkpointer=strict).aget_state(thread)
        assert saved.values == {"canary": {"x": 7}}
        assert BUILT == []
    finally:
        await strict.adelete_thread(thread_id)
