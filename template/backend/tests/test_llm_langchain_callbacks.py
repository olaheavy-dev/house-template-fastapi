"""LangChain's BudgetGuard (#33 T3) is a thin layer over the core's Meter: it reads LangChain's
run, messages, parameters, usage and errors and hands the Meter neutral values. Here the Meter is
replaced by a recorder, so no database is involved."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.exceptions import ModelRateLimitError, ModelTimeoutError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.llm.core.bounds import request_bound
from app.llm.core.budget import CallTags, Meter
from app.llm.core.errors import FailureKind
from app.llm.core.prices import Price, Rates
from app.llm.core.usage import Usage
from app.llm.langchain.callbacks import BudgetGuard, input_bound

pytestmark = pytest.mark.anyio

SONNET = Price(
    Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000),
    checked="2026-11-06",
    source="https://example.com/pricing",
)
HELLO: list[BaseMessage] = [HumanMessage("hello")]


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, uuid.UUID, dict[str, Any]]] = []

    async def reserve(self, call_id: uuid.UUID, **given: Any) -> int:
        self.calls.append(("reserve", call_id, given))
        return 0

    async def settle(self, call_id: uuid.UUID, usage: Usage | None) -> None:
        self.calls.append(("settle", call_id, {"usage": usage}))

    async def failed(self, call_id: uuid.UUID, kind: FailureKind | None, *, streamed: bool) -> None:
        self.calls.append(("failed", call_id, {"kind": kind, "streamed": streamed}))


def guard() -> tuple[BudgetGuard, Recorder]:
    g = BudgetGuard(
        model="anthropic:pin",
        price=SONNET,
        cap_micro_usd=5,
        max_tokens=500,
        sessionmaker=lambda: None,  # type: ignore[arg-type,return-value]
    )
    recorder = Recorder()
    g.meter = recorder  # type: ignore[assignment]
    return g, recorder


def test_the_guards_settings_are_its_meters() -> None:
    g = BudgetGuard(
        model="anthropic:pin",
        price=SONNET,
        cap_micro_usd=5,
        max_tokens=500,
        user_cap_micro_usd=2,
        sessionmaker=lambda: None,  # type: ignore[arg-type,return-value]
    )
    assert isinstance(g.meter, Meter)
    assert (g.model, g.price, g.cap, g.user_cap, g.max_tokens) == (
        "anthropic:pin",
        SONNET,
        5,
        2,
        500,
    )
    g.cap, g.user_cap = 7, None
    assert (g.meter.cap, g.meter.user_cap) == (7, None)


def test_the_input_bound_is_the_core_bound_of_the_serialised_messages_and_params() -> None:
    params = {"tools": [{"type": "function"}]}
    expected = request_bound([[m.model_dump() for m in HELLO], params], [m.content for m in HELLO])
    assert input_bound(HELLO, params) == expected


async def test_a_call_start_reserves_the_bounds_with_the_metadata_as_tags() -> None:
    g, recorder = guard()
    run = uuid.uuid4()
    metadata = {"prompt": "ask", "prompt_sha": "abc", "feature": "answer", "user_ref": "u_1"}
    params = {"max_completion_tokens": 700}
    await g.on_chat_model_start(
        {}, [HELLO, HELLO], run_id=run, metadata=metadata, invocation_params=params
    )
    [(event, call_id, given)] = recorder.calls
    assert (event, call_id) == ("reserve", run)
    # Two batches: twice one batch's bound; the bound limit 700 is larger than the gateway's 500.
    assert given == {
        "input_tokens": 2 * input_bound(HELLO, params),
        "output_tokens": 700,
        "tags": CallTags(prompt="ask", prompt_sha="abc", feature="answer", user_ref="u_1"),
    }


async def test_the_end_of_a_call_settles_its_usage_as_the_neutral_usage() -> None:
    g, recorder = guard()
    with_usage, without, empty = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    usage: Any = {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "output_token_details": {"reasoning": 3},
    }

    def end(message: AIMessage) -> LLMResult:
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    await g.on_llm_end(end(AIMessage("ok", usage_metadata=usage)), run_id=with_usage)
    await g.on_llm_end(end(AIMessage("ok")), run_id=without)
    await g.on_llm_end(LLMResult(generations=[[]]), run_id=empty)
    assert recorder.calls == [
        (
            "settle",
            with_usage,
            {"usage": Usage(input_tokens=10, output_tokens=4, reasoning_tokens=3)},
        ),
        ("settle", without, {"usage": None}),
        ("settle", empty, {"usage": None}),
    ]


async def test_a_failure_hands_the_meter_its_kind_and_whether_it_streamed() -> None:
    g, recorder = guard()
    streamed, plain, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await g.on_llm_new_token("Fo", run_id=streamed)
    await g.on_llm_error(ModelRateLimitError("x"), run_id=streamed)
    await g.on_llm_error(ModelTimeoutError("x"), run_id=plain)
    await g.on_llm_error(KeyError("x"), run_id=other)
    assert recorder.calls == [
        ("failed", streamed, {"kind": FailureKind.RATE_LIMITED, "streamed": True}),
        ("failed", plain, {"kind": FailureKind.TIMED_OUT, "streamed": False}),
        ("failed", other, {"kind": None, "streamed": False}),
    ]
    # The streaming mark is dropped once the call ends: the same id failing again is not streamed.
    await g.on_llm_error(ModelRateLimitError("x"), run_id=streamed)
    assert recorder.calls[-1][2] == {"kind": FailureKind.RATE_LIMITED, "streamed": False}
