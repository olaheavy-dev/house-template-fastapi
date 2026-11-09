"""The agent step limit through the LangChain adapter (#34).

An agent makes at most N model calls per run. Calls 1 to N-1 are offered the tools; call N is made
without tools and is the final answer, and the result carries `step_limit_reached`.

Every expected count is worked out by hand from that rule, not from running the code. With N = 3
and a model that asks for the `add` tool whenever it is offered one:
  call 1 asks for add, add runs      -> messages: human, ai, tool            (3)
  call 2 asks for add, add runs      -> messages: + ai, tool                  (5)
  call 3 is offered nothing, answers -> messages: + ai                        (6)
so 3 model calls, 2 tool runs, 6 messages, 3 llm_call rows. For N = 13: 13 calls, 12 tool runs,
1 + 12 x 2 + 1 = 26 messages. Each call costs 15 micro-USD: 100 input tokens x 0.10 + 10 output
tokens x 0.50 at gpt-6-luna's prices ($0.10 and $0.50 per million tokens).
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.exceptions import ModelRateLimitError, ModelTimeoutError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall, LlmSpendDay, expire_stale, utc_today
from app.llm.core.steps import StepLimit
from app.llm.langchain import steps
from app.llm.langchain.gateway import Gateway
from tests.test_llm_fakes import LoopingFake

pytestmark = pytest.mark.anyio

COST_PER_CALL = 15  # micro-USD, see the module docstring


@pytest.fixture(params=["langchain"])
def framework(request: pytest.FixtureRequest) -> str:
    """The framework under test. The MAF adapter (#35 5b) adds "maf" here, so the same tests prove
    both frameworks give the same answer at the limit."""
    return str(request.param)


def build_agent(
    framework: str, gateway: Gateway, tools: list[Callable[..., Any]], **options: Any
) -> Any:
    assert framework == "langchain"
    return gateway.agent(tools, **options)


def flag_of(framework: str, state: dict[str, Any]) -> bool:
    assert framework == "langchain"
    return steps.step_limit_reached(state)


@pytest.fixture
async def gateway(app: FastAPI, db_client: httpx.AsyncClient) -> Gateway:
    gateway: Gateway = app.state.gateway
    return gateway


@pytest.fixture
async def thread_id(gateway: Gateway) -> AsyncIterator[str]:
    thread = f"test-{uuid.uuid4()}"
    yield thread
    assert gateway.checkpointer is not None
    await gateway.checkpointer.adelete_thread(thread)


class Setup:
    """A gateway whose model is a LoopingFake, and the tool runs its agents make."""

    def __init__(self, gateway: Gateway, limit: int, **fake: Any) -> None:
        gateway.step_limit = StepLimit(limit)
        self.fake = LoopingFake(callbacks=[gateway.guard], **fake)
        gateway._model = self.fake
        self.tool_runs: list[tuple[int, int]] = []

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            self.tool_runs.append((a, b))
            return a + b

        self.add = add


def config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def llm_calls(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).order_by(LlmCall.created_at))).all())


async def committed(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == utc_today())
    )


def reached(logs: list[Any]) -> list[Any]:
    return [entry for entry in logs if entry["event"] == "step_limit_reached"]


# --- the final call is made without tools, and answers ------------------------------------------


@pytest.mark.parametrize(
    ("limit", "tool_runs", "messages"),
    [(3, 2, 6), (13, 12, 26)],  # see the module docstring for the working
)
async def test_the_last_call_is_made_without_tools_and_answers(
    framework: str,
    gateway: Gateway,
    thread_id: str,
    db_session: AsyncSession,
    limit: int,
    tool_runs: int,
    messages: int,
) -> None:
    setup = Setup(gateway, limit)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        state = await agent.ainvoke({"messages": [HumanMessage("Add forever")]}, config(thread_id))

    assert len(setup.fake.offered) == limit
    # Calls 1 to N-1 were offered add; call N was offered nothing.
    assert setup.fake.offered == [["add"]] * (limit - 1) + [[]]
    assert len(setup.tool_runs) == tool_runs
    assert len(state["messages"]) == messages
    last = state["messages"][-1]
    assert (last.type, last.text, last.tool_calls) == ("ai", "final", [])
    assert flag_of(framework, state) is True
    assert len(reached(logs)) == 1
    assert reached(logs)[0]["log_level"] == "warning"
    rows = await llm_calls(db_session)
    assert [row.status for row in rows] == ["settled"] * limit
    assert await committed(db_session) == COST_PER_CALL * limit


async def test_an_agent_that_finishes_before_the_limit_has_no_flag_and_no_log(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # N = 3, the model answers at call 2: 2 calls, 1 tool run, human + ai + tool + ai = 4.
    setup = Setup(gateway, 3, answer_on=2)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        state = await agent.ainvoke({"messages": [HumanMessage("Add once")]}, config(thread_id))
    assert len(setup.fake.offered) == 2
    assert len(setup.tool_runs) == 1
    assert len(state["messages"]) == 4
    assert flag_of(framework, state) is False
    assert "step_limit_reached" not in state
    assert reached(logs) == []
    assert len(await llm_calls(db_session)) == 2


async def test_a_model_that_would_answer_at_the_last_call_anyway_is_still_flagged(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    # N = 3, the model would answer at call 3: still 3 calls, 2 tool runs, 6 messages; the limit
    # was reached because call 3 was the last one allowed.
    setup = Setup(gateway, 3, answer_on=3)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert (len(setup.fake.offered), len(setup.tool_runs), len(state["messages"])) == (3, 2, 6)
    assert flag_of(framework, state) is True
    assert len(reached(logs)) == 1


async def test_one_allowed_call_is_the_final_answer_with_no_tools(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # N = 1: the only call is the last one. 1 call, 0 tool runs, human + ai = 2.
    setup = Setup(gateway, 1)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert setup.fake.offered == [[]]
    assert (len(setup.tool_runs), len(state["messages"])) == (0, 2)
    assert flag_of(framework, state) is True
    assert len(reached(logs)) == 1
    assert len(await llm_calls(db_session)) == 1


async def test_parallel_tool_calls_in_one_turn_count_as_one_model_call(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # N = 3, two add calls per turn: 3 model calls, 2 turns x 2 = 4 tool runs, messages =
    # human + (ai + 2 tool) x 2 + final ai = 1 + 6 + 1 = 8.
    setup = Setup(gateway, 3, parallel=2)
    agent = build_agent(framework, gateway, [setup.add])
    state = await agent.ainvoke({"messages": [HumanMessage("Add twice")]}, config(thread_id))
    assert len(setup.fake.offered) == 3
    assert len(setup.tool_runs) == 4
    assert len(state["messages"]) == 8
    assert flag_of(framework, state) is True
    assert len(await llm_calls(db_session)) == 3


async def test_a_model_that_ignores_the_empty_tool_list_still_stops_at_the_limit(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # The model asks for add even when offered nothing. Call 3 would end the loop only if its
    # tool calls are removed: 3 calls, 2 tool runs, 6 messages, the last without tool calls.
    setup = Setup(gateway, 3, ignore_unbound=True)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert (len(setup.fake.offered), len(setup.tool_runs), len(state["messages"])) == (3, 2, 6)
    last = state["messages"][-1]
    assert (last.type, last.tool_calls, last.invalid_tool_calls) == ("ai", [], [])
    assert flag_of(framework, state) is True
    assert len(reached(logs)) == 1
    assert len(await llm_calls(db_session)) == 3


async def test_the_count_restarts_on_each_run_of_the_same_thread(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # Two runs of N = 3 on one thread: 6 calls, 4 tool runs. Messages: the first run leaves 6,
    # the second adds human + 2 x (ai + tool) + final ai = 1 + 4 + 1 = 6, so 12.
    setup = Setup(gateway, 3)
    agent = build_agent(framework, gateway, [setup.add])
    with capture_logs() as logs:
        first = await agent.ainvoke({"messages": [HumanMessage("One")]}, config(thread_id))
        second = await agent.ainvoke({"messages": [HumanMessage("Two")]}, config(thread_id))
    assert len(first["messages"]) == 6
    assert len(second["messages"]) == 12
    assert len(setup.fake.offered) == 6
    # The second run's first call (the fourth overall) was offered the tool again.
    assert setup.fake.offered == [["add"], ["add"], [], ["add"], ["add"], []]
    assert len(setup.tool_runs) == 4
    assert flag_of(framework, second) is True
    assert len(reached(logs)) == 2
    assert len(await llm_calls(db_session)) == 6


async def test_the_limit_is_read_from_the_gateways_step_limit_at_each_agent(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    setup = Setup(gateway, 2)
    agent = build_agent(framework, gateway, [setup.add])
    await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert setup.fake.offered == [["add"], []]


# --- the caller's middleware --------------------------------------------------------------------


class AlwaysOffer(AgentMiddleware):
    """A caller's middleware that sets the tools on every call, as one that adds its own would."""

    def __init__(self, tool: StructuredTool) -> None:
        self.tool = tool

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(tools=[self.tool]))


async def test_a_callers_middleware_cannot_add_tools_back_on_the_last_call(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    setup = Setup(gateway, 3)
    tool = StructuredTool.from_function(setup.add)
    agent = build_agent(framework, gateway, [setup.add], middleware=[AlwaysOffer(tool)])
    state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert setup.fake.offered == [["add"], ["add"], []]
    assert len(state["messages"]) == 6
    assert flag_of(framework, state) is True


class Hooks(AgentMiddleware):
    """A caller's middleware with a node before and after every model call."""

    def before_model(self, state: Any, runtime: Any) -> None:
        return None

    def after_model(self, state: Any, runtime: Any) -> None:
        return None


class AsyncHooks(AgentMiddleware):
    async def abefore_model(self, state: Any, runtime: Any) -> None:
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        return None


async def test_a_callers_model_hooks_do_not_trip_the_recursion_backstop(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    # N = 13 with before_model and after_model hooks takes 52 steps (see the step-count tests
    # below), more than the 29 the backstop allows without hooks; with two hooks it is
    # (2 + 2) x 13 + 3 = 55.
    setup = Setup(gateway, 13)
    agent = build_agent(framework, gateway, [setup.add], middleware=[Hooks()])
    state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert len(setup.fake.offered) == 13
    assert flag_of(framework, state) is True


def test_the_backstop_is_two_plus_the_hooks_times_the_limit_plus_three() -> None:
    # No hooks: 2 x 13 + 3 = 29. One middleware with before_model and after_model: h = 2 and
    # (2 + 2) x 13 + 3 = 55; each hook of one middleware counts, sync or async.
    limit = StepLimit(13)
    assert steps.recursion_limit(limit, []) == 29
    assert steps.recursion_limit(limit, [Hooks()]) == 55
    assert steps.recursion_limit(limit, [AsyncHooks()]) == 55
    assert steps.recursion_limit(limit, [Hooks(), AsyncHooks()]) == 29 + 13 * 4  # h = 4
    # A middleware with no model hooks adds nothing: N = 3 gives 2 x 3 + 3 = 9.
    assert steps.recursion_limit(StepLimit(3), [AgentMiddleware()]) == 9


# --- structured output, and sync ----------------------------------------------------------------


class Answer(BaseModel):
    total: int


async def test_a_structured_answer_survives_the_last_call(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    # N = 2 with a ToolStrategy: call 1 asks for add, call 2 is offered no tool of ours but still
    # the structured-output tool, which the model calls. Its tool call is the answer, so it is not
    # removed.
    setup = Setup(gateway, 2, structured="Answer", structured_args={"total": 2})
    agent = build_agent(
        framework, gateway, [setup.add], response_format=ToolStrategy(schema=Answer)
    )
    state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    assert setup.fake.offered[0] == ["add", "Answer"]
    assert setup.fake.offered[1] == ["Answer"]
    assert state["structured_response"] == Answer(total=2)
    assert flag_of(framework, state) is True


def test_a_synchronous_run_gives_the_same_answer_at_the_limit() -> None:
    # No gateway, no database: the middleware on its own, run with invoke(). Same counts as the
    # N = 3 case in the module docstring.
    runs: list[int] = []

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        runs.append(a + b)
        return a + b

    fake = LoopingFake()
    agent = create_agent(fake, [add], middleware=[steps.StepLimitMiddleware(StepLimit(3))])
    with capture_logs() as logs:
        state = agent.invoke({"messages": [HumanMessage("Add")]})
    assert fake.offered == [["add"], ["add"], []]
    assert (len(runs), len(state["messages"])) == (2, 6)
    assert steps.step_limit_reached(state) is True
    assert len(reached(logs)) == 1


def test_the_flag_is_false_for_a_result_without_it() -> None:
    assert steps.step_limit_reached({"messages": []}) is False
    assert steps.step_limit_reached({"step_limit_reached": True}) is True


# --- one route, end to end ----------------------------------------------------------------------


async def test_a_looping_agent_behind_a_route_answers_200_not_422(
    app: FastAPI, gateway: Gateway, thread_id: str, db_client: httpx.AsyncClient
) -> None:
    setup = Setup(gateway, 3)

    @app.post("/loop")
    async def loop() -> dict[str, Any]:
        agent = gateway.agent([setup.add])
        state = await agent.ainvoke({"messages": [HumanMessage("Add forever")]}, config(thread_id))
        return {"text": state["messages"][-1].text, "cut_short": steps.step_limit_reached(state)}

    response = await db_client.post("/loop")
    assert response.status_code == 200
    assert response.json() == {"text": "final", "cut_short": True}


async def test_a_recursion_error_from_a_graph_is_a_plain_500(
    app: FastAPI, db_client: httpx.AsyncClient
) -> None:
    @app.post("/recursion")
    async def recursion() -> None:
        raise GraphRecursionError("recursion")

    response = await db_client.post("/recursion")
    assert response.status_code == 500


# --- thread, budget and failures on the final call (T4) -----------------------------------------


async def test_the_thread_is_saved_after_the_forced_answer_without_the_flag(
    framework: str, gateway: Gateway, thread_id: str
) -> None:
    setup = Setup(gateway, 3)
    agent = build_agent(framework, gateway, [setup.add])
    await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    saved = await agent.aget_state(config(thread_id))
    assert len(saved.values["messages"]) == 6
    assert "step_limit_reached" not in saved.values
    assert "run_model_calls" not in saved.values


async def test_the_final_call_has_its_own_reservation_and_row(
    framework: str, gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    setup = Setup(gateway, 3)
    agent = build_agent(framework, gateway, [setup.add])
    await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    rows = await llm_calls(db_session)
    assert len({row.id for row in rows}) == 3
    assert [row.status for row in rows] == ["settled"] * 3
    assert [row.cost_micro_usd for row in rows] == [COST_PER_CALL] * 3
    assert all(row.reserved_micro_usd >= COST_PER_CALL for row in rows)


def route(app: FastAPI, gateway: Gateway, setup: Setup, thread_id: str) -> None:
    @app.post("/ask")
    async def ask() -> dict[str, Any]:
        agent = gateway.agent([setup.add])
        state = await agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
        return {"cut_short": steps.step_limit_reached(state)}


async def test_a_final_call_that_times_out_keeps_its_reservation(
    app: FastAPI,
    gateway: Gateway,
    thread_id: str,
    db_client: httpx.AsyncClient,
    db_session: AsyncSession,
) -> None:
    setup = Setup(gateway, 3, raises={3: ModelTimeoutError("timed out")})
    route(app, gateway, setup, thread_id)
    with capture_logs() as logs:
        response = await db_client.post("/ask")
    assert response.status_code == 504
    assert reached(logs) == []
    # Two settled calls and the failed third, which may have been billed: it keeps its
    # reservation as its cost. The thread holds what was saved before it: human, ai, tool, ai, tool.
    first, second, third = await llm_calls(db_session)
    assert (first.status, second.status, third.status) == ("settled", "settled", "failed")
    assert third.cost_micro_usd == third.reserved_micro_usd
    assert await committed(db_session) == 2 * COST_PER_CALL + third.reserved_micro_usd
    agent = gateway.agent([setup.add])
    saved = await agent.aget_state(config(thread_id))
    assert len(saved.values["messages"]) == 5
    assert "step_limit_reached" not in saved.values


async def test_a_final_call_the_provider_rejects_releases_its_reservation(
    app: FastAPI,
    gateway: Gateway,
    thread_id: str,
    db_client: httpx.AsyncClient,
    db_session: AsyncSession,
) -> None:
    setup = Setup(gateway, 3, raises={3: ModelRateLimitError("slow down")})
    route(app, gateway, setup, thread_id)
    response = await db_client.post("/ask")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    # A rate limit is answered before any work is done: the third call settles at zero.
    *_, third = await llm_calls(db_session)
    assert (third.status, third.cost_micro_usd) == ("failed", 0)
    assert await committed(db_session) == 2 * COST_PER_CALL


async def test_a_cancelled_final_call_keeps_its_reservation(
    gateway: Gateway, thread_id: str, db_session: AsyncSession
) -> None:
    # The third call blocks until released; the task is cancelled while it waits. Whether it was
    # billed is unknown, so its reservation stays counted.
    release = threading.Event()
    entered = threading.Event()

    def block() -> None:
        entered.set()
        release.wait(5)

    setup = Setup(gateway, 3, hooks={3: block})
    agent = build_agent("langchain", gateway, [setup.add])
    task = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("Add")]}, config(thread_id))
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Left "reserved", counted at its reservation, until expire_stale closes it at that cost.
    first, second, third = await llm_calls(db_session)
    assert (first.status, second.status, third.status) == ("settled", "settled", "reserved")
    assert third.cost_micro_usd is None
    assert await committed(db_session) == 2 * COST_PER_CALL + third.reserved_micro_usd
    await db_session.execute(
        update(LlmCall)
        .where(LlmCall.id == third.id)
        .values(created_at=func.now() - timedelta(hours=2))
    )
    assert await expire_stale(db_session, older_than=timedelta(minutes=30)) == 1
    closed = await db_session.get(LlmCall, third.id, populate_existing=True)
    assert closed is not None
    assert (closed.status, closed.cost_micro_usd) == ("failed", closed.reserved_micro_usd)
    assert await committed(db_session) == 2 * COST_PER_CALL + closed.reserved_micro_usd


async def test_a_budget_refusal_on_the_final_call_is_a_503_and_reserves_nothing(
    app: FastAPI,
    gateway: Gateway,
    thread_id: str,
    db_client: httpx.AsyncClient,
    db_session: AsyncSession,
) -> None:
    # While call 2 runs the daily cap drops to 1 micro-USD, so call 3's reservation would cross it.
    def lower_cap() -> None:
        gateway.guard.cap = 1

    setup = Setup(gateway, 3, hooks={2: lower_cap})
    route(app, gateway, setup, thread_id)
    response = await db_client.post("/ask")
    assert response.status_code == 503
    rows = await llm_calls(db_session)
    assert len(rows) == 2  # the refused third call reserved nothing
    assert [row.status for row in rows] == ["settled", "settled"]
    assert await committed(db_session) == 2 * COST_PER_CALL


# --- how many LangGraph steps a run takes (proves the backstop's arithmetic) --------------------


def run_with_recursion_limit(limit: int, steps_allowed: int, *middleware: Any) -> dict[str, Any]:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = create_agent(
        LoopingFake(), [add], middleware=[*middleware, steps.StepLimitMiddleware(StepLimit(limit))]
    ).with_config(recursion_limit=steps_allowed)
    result: dict[str, Any] = agent.invoke({"messages": [HumanMessage("Add")]})
    return result


def test_a_run_of_thirteen_calls_takes_26_steps_and_no_fewer() -> None:
    # Measured with the pinned LangGraph (the plan said 2N - 1 = 25): 13 model steps and 12 tool
    # steps make 25, and LangGraph's recursion limit counts one more, so a run takes 2N = 26.
    assert steps.step_limit_reached(run_with_recursion_limit(13, 26))
    with pytest.raises(GraphRecursionError):
        run_with_recursion_limit(13, 25)


def test_a_run_with_before_and_after_model_hooks_takes_52_steps_and_no_fewer() -> None:
    # Each hook node adds one step per model call: (2 + 2) x 13 = 52, the same one more than
    # 12 x 4 + 3 = 51. The backstop for h = 2 is 55, so it never stops this run.
    assert steps.step_limit_reached(run_with_recursion_limit(13, 52, Hooks()))
    with pytest.raises(GraphRecursionError):
        run_with_recursion_limit(13, 51, Hooks())
