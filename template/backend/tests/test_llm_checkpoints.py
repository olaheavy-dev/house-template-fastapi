"""Agent threads saved in Postgres: what the checkpointer can read back (#36)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.llm.langchain.gateway import Gateway

pytestmark = pytest.mark.anyio

USAGE: UsageMetadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


class ToolCallingFake(GenericFakeChatModel):
    """Answers from a script, and accepts tools as create_agent binds them."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


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
