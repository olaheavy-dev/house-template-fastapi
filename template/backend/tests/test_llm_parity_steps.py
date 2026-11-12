"""The agent step limit, the same under every framework (decision 0003, #34 and #35): a run makes at
most AGENT_MAX_MODEL_CALLS model calls, the last one is made without tools so the model writes a
final answer, the result carries the neutral flag, and the run is logged once as
`step_limit_reached`. The database is real; the model is scripted by the harness (tests/harness/),
so each test runs once per framework the project chose and imports none.

The counts are worked out by hand from that rule. The scripted model asks for the `add` tool
whenever it is offered one and answers "final answer" when it is not. With a limit of N:
  calls 1 to N-1 are offered the tool, ask for it, and `add` runs after each: N-1 tool runs;
  call N is offered nothing and answers;
so N model calls, N-1 tool runs, N llm_call rows (each call is reserved and settled on its own).
For N = 1 that is one call, no tool run; for the default 13, 13 calls and 12 tool runs.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall
from app.settings import Settings
from tests.harness import AsksTool, Harness, Reply

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 9)
MODEL = "anthropic:claude-sonnet-5-5"
LOOPS = AsksTool("add", {"a": 1, "b": 2})
QUESTION = "What is 1 + 2?"


@pytest.fixture
def make(framework: Harness, settings: Settings, db_session: AsyncSession) -> Callable[[int], Any]:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def build(limit: int) -> Any:
        gateway = framework.gateway(
            settings.model_copy(update={"llm_model": MODEL, "agent_max_model_calls": limit}),
            lambda: sessions,
        )
        meter = framework.meter(gateway)
        meter.cap, meter.today = 1_000_000, lambda: DAY
        return gateway

    return build


class Adder:
    """The agent's one tool, recording each run."""

    def __init__(self) -> None:
        self.runs: list[tuple[int, int]] = []

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            self.runs.append((a, b))
            return a + b

        self.add = add


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(await session.scalars(select(LlmCall).where(LlmCall.day == DAY)))


def reached(logs: list[Any]) -> list[Any]:
    return [entry for entry in logs if entry["event"] == "step_limit_reached"]


@pytest.mark.parametrize(("limit", "tool_runs"), [(1, 0), (2, 1), (3, 2), (13, 12)])
async def test_a_looping_agent_makes_exactly_its_limit_of_calls_and_answers_without_tools(
    framework: Harness,
    make: Callable[[int], Any],
    db_session: AsyncSession,
    limit: int,
    tool_runs: int,
) -> None:
    gateway = make(limit)
    framework.install_scripted(gateway, [LOOPS] * (limit + 2))
    adder = Adder()
    with capture_logs() as logs:
        answer, flagged = await framework.run_agent(gateway, QUESTION, [adder.add])
    assert answer == "final answer"
    assert flagged
    assert len(adder.runs) == tool_runs
    assert framework.scripted_left(gateway) == 2  # exactly `limit` model calls, not one more
    assert not framework.last_request_offered_tools(gateway)
    calls = await rows(db_session)
    assert len(calls) == limit
    assert {call.status for call in calls} == {"settled"}
    assert [entry["max_model_calls"] for entry in reached(logs)] == [limit]


async def test_an_agent_that_answers_before_its_limit_is_not_flagged_or_logged(
    framework: Harness, make: Callable[[int], Any], db_session: AsyncSession
) -> None:
    gateway = make(3)
    framework.install_scripted(gateway, [LOOPS, Reply("done early")])
    adder = Adder()
    with capture_logs() as logs:
        answer, flagged = await framework.run_agent(gateway, QUESTION, [adder.add])
    assert (answer, flagged) == ("done early", False)
    assert len(adder.runs) == 1
    assert len(await rows(db_session)) == 2
    assert reached(logs) == []


async def test_the_count_restarts_on_every_run(
    framework: Harness, make: Callable[[int], Any], db_session: AsyncSession
) -> None:
    gateway = make(2)
    framework.install_scripted(gateway, [LOOPS] * 4)
    adder = Adder()
    first = await framework.run_agent(gateway, QUESTION, [adder.add])
    second = await framework.run_agent(gateway, QUESTION, [adder.add])
    assert first == second == ("final answer", True)
    assert len(adder.runs) == 2  # one per run
    assert len(await rows(db_session)) == 4


async def test_every_call_of_a_run_carries_the_runs_tags(
    framework: Harness, make: Callable[[int], Any], db_session: AsyncSession
) -> None:
    gateway = make(3)
    framework.install_scripted(gateway, [LOOPS] * 3)
    await framework.run_agent(
        gateway, QUESTION, [Adder().add], tags={"feature": "sums", "user_ref": "u_7"}
    )
    calls = await rows(db_session)
    assert {(call.feature, call.user_ref) for call in calls} == {("sums", "u_7")}
    assert len(calls) == 3
