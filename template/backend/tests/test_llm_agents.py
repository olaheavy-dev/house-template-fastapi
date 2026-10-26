from __future__ import annotations

import ast
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import checkpoints
from app.llm.budget import LlmCall
from app.llm.gateway import Gateway

pytestmark = pytest.mark.anyio

APP = Path(__file__).resolve().parents[1] / "src" / "app"
USAGE: UsageMetadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


class ToolCallingFake(GenericFakeChatModel):
    """Answers from a script, and accepts tools as create_agent binds them."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def test_checkpoint_tables_are_at_the_installed_langgraph_version(
    db_session: AsyncSession,
) -> None:
    # Fails after an upgrade that adds a checkpoint migration: write an Alembic revision whose
    # upgrade() calls checkpoints.apply_migrations inside autocommit_block(), like 0003.
    version = await db_session.scalar(text("SELECT max(v) FROM checkpoint_migrations"))
    assert version == len(checkpoints.MIGRATIONS) - 1


async def test_an_agent_runs_its_tools_and_keeps_its_thread(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway: Gateway = app.state.gateway
    script = [
        AIMessage(
            "",
            tool_calls=[{"name": "add", "args": {"a": 2, "b": 2}, "id": "call-1"}],
            usage_metadata=USAGE,
        ),
        AIMessage("2 + 2 is 4.", usage_metadata=USAGE),
    ]
    # The gateway's model, scripted: same budget guard, no provider.
    gateway._model = ToolCallingFake(messages=iter(script), callbacks=[gateway.guard])
    thread_id = f"test-{uuid.uuid4()}"
    thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    agent = gateway.agent([add], system_prompt="You add numbers.")
    try:
        state = await agent.ainvoke({"messages": [HumanMessage("What is 2 + 2?")]}, config=thread)
        assert state["messages"][-1].text == "2 + 2 is 4."
        assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "ai"]

        # Saved in Postgres: a later request on the same thread picks the conversation up.
        saved = await agent.aget_state(thread)
        assert len(saved.values["messages"]) == 4
    finally:
        assert gateway.checkpointer is not None
        await gateway.checkpointer.adelete_thread(thread_id)

    # Every model call in the loop went through the budget.
    statuses = (await db_session.scalars(select(LlmCall.status))).all()
    assert statuses == ["settled", "settled"]


def string_models(source: str) -> list[int]:
    """Lines that hand LangChain a "provider:model" string, which skips the gateway's budget."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in {"create_agent", "init_chat_model", "create_react_agent"}:
            continue
        model = node.args[0] if node.args else None
        model = next((k.value for k in node.keywords if k.arg == "model"), model)
        if isinstance(model, ast.Constant | ast.JoinedStr):
            lines.append(node.lineno)
    return lines


def test_no_agent_or_model_is_built_from_a_string() -> None:
    found = {
        str(path.relative_to(APP)): lines
        for path in APP.rglob("*.py")
        if (lines := string_models(path.read_text()))
    }
    assert found == {}


@pytest.mark.parametrize(
    "source",
    [
        'create_agent("anthropic:claude-haiku-5-5", tools)',
        'create_agent(model="openai:gpt-6-luna")',
        'agents.create_agent(f"anthropic:{name}")',
        'init_chat_model("anthropic:claude-haiku-5-5")',
    ],
)
def test_the_string_check_catches_each_form(source: str) -> None:
    assert string_models(source) == [1]


def test_the_string_check_allows_the_gateways_model() -> None:
    assert string_models("create_agent(gateway.model(), tools)") == []
    assert string_models("init_chat_model(settings.llm_model)") == []


async def test_without_a_database_the_checkpoint_pool_stays_empty(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    # min_size=0: the app starts (and /health/live answers) without opening a checkpoint
    # connection; one is made only when an agent runs.
    assert (await client.get("/health/live")).status_code == 200
    gateway: Gateway = app.state.gateway
    assert isinstance(gateway.checkpointer, AsyncPostgresSaver)
    pool = gateway.checkpointer.conn
    assert isinstance(pool, AsyncConnectionPool)
    assert pool.get_stats()["pool_size"] == 0
