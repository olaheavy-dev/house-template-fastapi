"""The agent step limit for MAF (decision 0003, the MAF half of #34): a run makes at most
AGENT_MAX_MODEL_CALLS model calls, the last one without tools, and its result says so.

A run is one `get_response` of the gateway's client: MAF's tool loop (FunctionInvocationLayer) makes
each model call of the run inside it, and an Agent's `run` is one such call. The gateway's client
puts a fresh StepRun into the run's options (RUN_OPTION); MAF copies the options once per run but
keeps the object, so every model call of the run sees the same StepRun (measured on
agent-framework-core 1.25.0). BudgetMiddleware takes it out of the options before anything reaches
the provider, counts the call with the core's StepLimit and, on the last allowed call, sends
`tool_choice="none"` and marks the run. When the run returns, its result carries the neutral flag
(`step_limit_reached`, read with step_limit_reached()) and the run is logged once.

MAF's own limit is the backstop: `max_iterations` is set to N - 1 (app/llm/maf/gateway.py), so MAF
also ends the run with a tools-off call at call N and never makes call N + 1, even if a model asks
for tools when told not to.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from typing import Any, cast

from agent_framework import ChatResponse, ChatResponseUpdate, ResponseStream

from app.llm.core.steps import STEP_LIMIT_REACHED, StepLimit, log_reached

__all__ = ["RUN_OPTION", "StepRun", "flagged", "step_limit_reached"]

# The option that carries a run's StepRun from the gateway's client to its budget middleware. It
# never reaches the provider: the middleware removes it first.
RUN_OPTION = "app.llm.step_run"


@dataclass(slots=True)
class StepRun:
    """One run's count of model calls. `limit` is set by the budget middleware (its StepLimit)."""

    calls: int = 0
    forced: bool = False
    limit: StepLimit | None = None


def step_limit_reached(result: Any) -> bool:
    """Whether a run's result (a ChatResponse, or an Agent's AgentResponse, which shares its
    additional properties) ended on its forced final answer: tell users it was cut short."""
    properties = getattr(result, "additional_properties", None)
    return isinstance(properties, dict) and properties.get(STEP_LIMIT_REACHED) is True


def _reached(run: StepRun) -> None:
    if run.limit is not None:
        log_reached(run.limit)


def flagged(result: Any, run: StepRun) -> Any:
    """`result` (what the client's get_response returned for the run) with the flag set and the
    run logged once, if its final call was forced. A stream gets the flag as one last update with
    no content, so an Agent's streamed response (built from updates) carries it too."""
    if isinstance(result, ResponseStream):
        stream = cast("ResponseStream[ChatResponseUpdate, ChatResponse[Any]]", result)

        async def updates() -> AsyncGenerator[ChatResponseUpdate]:
            async with stream:
                async for update in stream:
                    yield update
            if run.forced:
                _reached(run)
                yield ChatResponseUpdate(
                    role="assistant", contents=[], additional_properties={STEP_LIMIT_REACHED: True}
                )

        async def final(collected: Any) -> ChatResponse[Any]:
            response = await stream.get_final_response()
            if run.forced:
                response.additional_properties[STEP_LIMIT_REACHED] = True
            return response

        return ResponseStream(updates(), finalizer=final)

    async def answered() -> Any:
        response = await cast(Awaitable[Any], result)
        if run.forced:
            _reached(run)
            response.additional_properties[STEP_LIMIT_REACHED] = True
        return response

    return answered()
