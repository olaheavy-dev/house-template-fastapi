"""The MAF agent (#35 PR 4): gateway.agent() on the gateway's client, its step limit and its tool
calls under redaction. The shared step-limit rules are in test_llm_parity_steps.py; these are the
MAF-only cases. The values are synthetic."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from agent_framework import ChatResponse, Content, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall
from app.llm.maf.gateway import Gateway
from app.llm.maf.steps import RUN_OPTION, step_limit_reached
from app.settings import Settings
from tests.harness import AsksTool
from tests.harness.maf import HARNESS, ScriptedClient

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 10)
EMAIL = "priya.shah@example.com"
LOOPS = AsksTool("add", {"a": 1, "b": 2})


@pytest.fixture
def make(settings: Settings, db_session: AsyncSession) -> Any:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def build(limit: int = 3) -> Gateway:
        gateway = Gateway(
            settings.model_copy(
                update={"llm_model": "anthropic:claude-sonnet-5-5", "agent_max_model_calls": limit}
            ),
            lambda: sessions,
        )
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        return gateway

    return build


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(await session.scalars(select(LlmCall).where(LlmCall.day == DAY)))


def scripted(gateway: Gateway, *items: Any) -> ScriptedClient:
    HARNESS.install_scripted(gateway, list(items))
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    return client


async def test_a_streamed_agent_run_is_flagged_and_logged_once(make: Any) -> None:
    gateway = make(3)
    scripted(gateway, LOOPS, LOOPS, LOOPS)
    with capture_logs() as logs:
        stream = gateway.agent([add]).run("What is 1 + 2?", stream=True)
        async with stream:
            async for _update in stream:
                pass
            response = await stream.get_final_response()
    assert response.text == "final answer"
    assert step_limit_reached(response)
    assert [e["max_model_calls"] for e in logs if e["event"] == "step_limit_reached"] == [3]


async def test_the_runs_count_never_reaches_the_provider(make: Any) -> None:
    gateway = make(2)
    client = scripted(gateway, LOOPS, LOOPS)
    await gateway.agent([add]).run("What is 1 + 2?")
    assert len(client.seen_options) == 2
    assert all(RUN_OPTION not in options for options in client.seen_options)


async def test_a_model_that_ignores_the_last_calls_tool_choice_gets_no_extra_call(
    make: Any, db_session: AsyncSession
) -> None:
    # The backstop: MAF's own loop limit (max_iterations = N - 1) also ends the run at call N, so
    # a model that asks for a tool when told not to has it ignored, not run, and is not called
    # again.
    gateway = make(2)
    client = scripted(gateway)
    runs: list[int] = []

    def counted(a: int, b: int) -> int:
        """Add two numbers."""
        runs.append(a + b)
        return a + b

    async def always_asks(**_kwargs: Any) -> ChatResponse[Any]:
        call = Content.from_function_call(f"c{len(runs)}", "counted", arguments='{"a": 1, "b": 2}')
        return ChatResponse(messages=[Message("assistant", [call])])

    client._inner_get_response = lambda **kwargs: always_asks(**kwargs)  # type: ignore[method-assign]
    response = await gateway.agent([counted]).run("What is 1 + 2?")
    assert step_limit_reached(response)
    assert runs == [3]  # only call 1's request ran the tool
    assert len(await rows(db_session)) == 2


async def test_a_tool_gets_the_original_and_its_result_goes_back_as_a_placeholder(
    make: Any,
) -> None:
    # #78 tests 19 and 20 through an agent: the model asks with the placeholder, the tool runs
    # with the original, and the tool's result (which repeats the email) is redacted again on the
    # next model call; the caller's answer has the original.
    gateway = make(3)
    got: list[str] = []

    def lookup(address: str) -> str:
        """Look a contact up."""
        got.append(address)
        return json.dumps({"owner": address, "status": "active"})

    asks = ChatResponse(
        messages=[
            Message(
                "assistant",
                [
                    Content.from_function_call(
                        "c1", "lookup", arguments=json.dumps({"address": "[EMAIL_1]"})
                    )
                ],
            )
        ]
    )
    client = scripted(
        gateway, asks, ChatResponse(messages=[Message("assistant", ["[EMAIL_1] is active"])])
    )
    response = await gateway.agent([lookup]).run(f"Is {EMAIL} active?")
    assert got == [EMAIL]
    second = json.dumps([message.to_dict() for message in client.seen[1]], default=str)
    assert EMAIL not in second
    assert "[EMAIL_1]" in second
    assert response.text == f"{EMAIL} is active"
