"""Live checks that the gateway keeps its promises with the real provider, whichever framework the
project chose. The recorded tests replay one answer; these call the model, so they catch what a
recording can't: the provider changing its usage reporting, a model that stops honouring structured
output or tool calls, or a price-table mistake. They hold for any product; add the product's own
evals next to them."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langsmith import testing as t
from pydantic import BaseModel

from app.llm.core.budget import BudgetExhausted, LlmCall
from tests.evals.conftest import NewCalls
from tests.harness import Harness, user

pytestmark = [pytest.mark.anyio, pytest.mark.eval]


def settled_at_real_cost(call: LlmCall) -> bool:
    return (
        call.status == "settled"
        and bool(call.input_tokens and call.output_tokens)
        and 0 < (call.cost_micro_usd or 0) < call.reserved_micro_usd
    )


@pytest.mark.langsmith
async def test_a_call_is_settled_at_its_real_cost(
    framework: Harness, gateway: Any, new_calls: NewCalls
) -> None:
    prompt = "Reply with the single word: ready"
    t.log_inputs({"prompt": prompt, "framework": framework.name})
    answer = await framework.ask(gateway, [user(prompt)])
    [call] = await new_calls()
    t.log_outputs({"answer": answer, "cost_micro_usd": call.cost_micro_usd})
    t.log_feedback(key="settled_at_real_cost", score=settled_at_real_cost(call))
    assert "ready" in answer.lower()
    assert settled_at_real_cost(call), call.__dict__


@pytest.mark.langsmith
async def test_a_stream_is_settled_from_its_final_usage(
    framework: Harness, gateway: Any, new_calls: NewCalls
) -> None:
    # Without usage in the stream, the budget would keep the whole worst-case reservation.
    prompt = "Count from 1 to 5, separated by commas."
    t.log_inputs({"prompt": prompt, "framework": framework.name})
    text = "".join(await framework.stream(gateway, [user(prompt)]))
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
    framework: Harness, gateway: Any, new_calls: NewCalls
) -> None:
    question = "What is the capital of Japan?"
    t.log_inputs({"question": question, "framework": framework.name})
    t.log_reference_outputs({"city": "Tokyo"})
    capital = await framework.ask_structured(gateway, [user(question)], Capital)
    [call] = await new_calls()
    assert isinstance(capital, Capital)
    t.log_outputs(capital.model_dump())
    t.log_feedback(key="correct", score=capital.city.strip().lower() == "tokyo")
    assert capital.city.strip().lower() == "tokyo"
    assert settled_at_real_cost(call), call.__dict__


@pytest.mark.langsmith
async def test_an_agent_uses_its_tool(
    framework: Harness, gateway: Any, new_calls: NewCalls
) -> None:
    question = "Use the add tool to add 48213 and 90577, then tell me the result."
    t.log_inputs({"question": question, "framework": framework.name})
    t.log_reference_outputs({"answer": "138790"})
    results: list[int] = []

    def add(a: int, b: int) -> int:
        """Add two whole numbers exactly."""
        results.append(a + b)
        return a + b

    final, cut_short = await framework.run_agent(
        gateway, question, [add], instructions="Use your tools for arithmetic."
    )
    calls = await new_calls()
    t.log_outputs({"answer": final, "tool_results": results, "model_calls": len(calls)})
    t.log_feedback(key="correct", score="138790" in final.replace(",", ""))
    assert results == [138790]
    assert "138790" in final.replace(",", "")
    assert not cut_short
    assert len(calls) >= 2
    assert all(settled_at_real_cost(call) for call in calls), [c.__dict__ for c in calls]


@pytest.mark.langsmith
async def test_a_stored_conversation_is_remembered_after_a_restart(
    framework: Harness, make_gateway: Any, new_calls: NewCalls
) -> None:
    # The conversation is keyed by the user and the conversation id, derived on the server; a new
    # gateway (a restart) reads it back from Postgres and resends it. The first turn runs a tool
    # loop, so what is stored holds a tool call and its result as well as text.
    user_ref, conversation = f"eval-{uuid.uuid4()}", "codeword"
    t.log_inputs({"framework": framework.name})

    def add(a: int, b: int) -> int:
        """Add two whole numbers exactly."""
        return a + b

    first = await make_gateway()
    try:
        await framework.thread_turn(
            first,
            user_ref=user_ref,
            conversation_id=conversation,
            messages=[
                user(
                    "Use the add tool to add 48213 and 90577. Also remember this code word for "
                    "later: ORCHID. Reply with the sum only."
                )
            ],
            tools=[add],
        )
        restarted = await make_gateway()
        answer = await framework.thread_turn(
            restarted,
            user_ref=user_ref,
            conversation_id=conversation,
            messages=[
                user(
                    "What was the code word I gave you, and what sum did the tool return? Reply "
                    "with the word and the number only."
                )
            ],
        )
    finally:
        await framework.forget_threads(first, user_ref=user_ref, conversation_id=conversation)
    calls = await new_calls()
    t.log_outputs({"answer": answer, "model_calls": len(calls)})
    remembered = "ORCHID" in answer.upper() and "138790" in answer.replace(",", "")
    t.log_feedback(key="remembered", score=remembered)
    assert "ORCHID" in answer.upper()
    assert "138790" in answer.replace(",", "")
    assert len(calls) >= 3  # the tool loop's two calls and the second turn
    assert all(settled_at_real_cost(call) for call in calls), [c.__dict__ for c in calls]


@pytest.mark.langsmith
async def test_the_budget_refuses_a_call_before_sending_it(
    framework: Harness, gateway: Any, new_calls: NewCalls
) -> None:
    framework.meter(gateway).cap = 1  # micro-USD: no call fits
    t.log_inputs({"cap_micro_usd": 1, "framework": framework.name})
    with pytest.raises(BudgetExhausted):
        await framework.ask(gateway, [user("This must never be sent.")])
    calls = await new_calls()
    t.log_outputs({"rows": len(calls)})
    t.log_feedback(key="refused_without_cost", score=calls == [])
    assert calls == []
