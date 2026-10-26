"""Live checks that the gateway keeps its promises with the real provider. The recorded tests
replay one answer; these call the model, so they catch what a recording can't: the provider
changing its usage reporting, a model that stops honouring structured output or tool calls, or a
price-table mistake. They hold for any product; add the product's own evals next to them."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langsmith import testing as t
from pydantic import BaseModel
from sqlalchemy import select

from app.llm.budget import BudgetExhausted, LlmCall
from app.llm.gateway import Gateway

pytestmark = [pytest.mark.anyio, pytest.mark.eval]

NewCalls = Callable[[], Awaitable[list[LlmCall]]]


@pytest.fixture
async def new_calls(gateway: Gateway) -> NewCalls:
    """The llm_call rows written since the test started (the database is shared by the run)."""
    sessions = gateway.guard.sessionmaker()
    async with sessions() as session:
        before = set((await session.scalars(select(LlmCall.id))).all())

    async def read() -> list[LlmCall]:
        async with sessions() as session:
            rows = (await session.scalars(select(LlmCall).order_by(LlmCall.created_at))).all()
        return [row for row in rows if row.id not in before]

    return read


def settled_at_real_cost(call: LlmCall) -> bool:
    return (
        call.status == "settled"
        and bool(call.input_tokens and call.output_tokens)
        and 0 < (call.cost_micro_usd or 0) < call.reserved_micro_usd
    )


@pytest.mark.langsmith
async def test_a_call_is_settled_at_its_real_cost(gateway: Gateway, new_calls: NewCalls) -> None:
    t.log_inputs({"prompt": "Reply with the single word: ready"})
    answer = await gateway.model().ainvoke([HumanMessage("Reply with the single word: ready")])
    [call] = await new_calls()
    t.log_outputs({"answer": answer.text, "cost_micro_usd": call.cost_micro_usd})
    t.log_feedback(key="settled_at_real_cost", score=settled_at_real_cost(call))
    assert "ready" in answer.text.lower()
    assert settled_at_real_cost(call), call.__dict__


@pytest.mark.langsmith
async def test_a_stream_is_settled_from_its_final_usage(
    gateway: Gateway, new_calls: NewCalls
) -> None:
    # Without usage in the stream, the guard would keep the whole worst-case reservation.
    prompt = "Count from 1 to 5, separated by commas."
    t.log_inputs({"prompt": prompt})
    text = "".join([chunk.text async for chunk in gateway.model().astream([HumanMessage(prompt)])])
    [call] = await new_calls()
    t.log_outputs({"answer": text, "cost_micro_usd": call.cost_micro_usd})
    t.log_feedback(key="settled_at_real_cost", score=settled_at_real_cost(call))
    assert "5" in text
    assert settled_at_real_cost(call), call.__dict__


class Capital(BaseModel):
    country: str
    city: str


@pytest.mark.langsmith
async def test_structured_output_is_valid_and_budgeted(
    gateway: Gateway, new_calls: NewCalls
) -> None:
    t.log_inputs({"question": "What is the capital of Japan?"})
    t.log_reference_outputs({"city": "Tokyo"})
    capital = (
        await gateway.model()
        .with_structured_output(Capital)
        .ainvoke([HumanMessage("What is the capital of Japan?")])
    )
    [call] = await new_calls()
    assert isinstance(capital, Capital)
    t.log_outputs(capital.model_dump())
    t.log_feedback(key="correct", score=capital.city.strip().lower() == "tokyo")
    assert capital.city.strip().lower() == "tokyo"
    assert settled_at_real_cost(call), call.__dict__


def add(a: int, b: int) -> int:
    """Add two whole numbers exactly."""
    return a + b


@pytest.mark.langsmith
async def test_an_agent_uses_its_tool_and_keeps_its_thread(
    gateway: Gateway, new_calls: NewCalls
) -> None:
    question = "Use the add tool to add 48213 and 90577, then tell me the result."
    t.log_inputs({"question": question})
    t.log_reference_outputs({"answer": "138790"})
    agent = gateway.agent([add], system_prompt="Use your tools for arithmetic.")
    thread_id = f"eval-{uuid.uuid4()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = await agent.ainvoke({"messages": [HumanMessage(question)]}, config=config)
        saved = await agent.aget_state(config)
    finally:
        assert gateway.checkpointer is not None
        await gateway.checkpointer.adelete_thread(thread_id)
    tool_results = [m.text for m in state["messages"] if isinstance(m, ToolMessage)]
    final = state["messages"][-1].text
    calls = await new_calls()
    t.log_outputs({"answer": final, "tool_results": tool_results, "model_calls": len(calls)})
    t.log_feedback(key="correct", score="138790" in final.replace(",", ""))
    assert tool_results == ["138790"]
    assert "138790" in final.replace(",", "")
    assert len(saved.values["messages"]) == len(state["messages"])
    assert len(calls) >= 2
    assert all(settled_at_real_cost(call) for call in calls), [c.__dict__ for c in calls]


@pytest.mark.langsmith
# With LangSmith tracing on, the refusal stops LangChain before it awaits the tracer's own
# chat-model-start callback, and Python warns that coroutine was never awaited. Harmless (the call
# is still refused and nothing is sent), so only that warning is allowed here.
@pytest.mark.filterwarnings("ignore:coroutine 'ahandle_event' was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_the_budget_refuses_a_call_before_sending_it(
    gateway: Gateway, new_calls: NewCalls
) -> None:
    gateway.guard.cap = 1  # micro-USD: no call fits
    t.log_inputs({"cap_micro_usd": 1})
    with pytest.raises(BudgetExhausted):
        await gateway.model().ainvoke([HumanMessage("This must never be sent.")])
    calls = await new_calls()
    t.log_outputs({"rows": len(calls)})
    t.log_feedback(key="refused_without_cost", score=calls == [])
    assert calls == []
