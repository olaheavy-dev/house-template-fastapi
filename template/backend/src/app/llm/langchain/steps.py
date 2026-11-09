"""The agent step limit for LangChain's create_agent (the rule is app/llm/core/steps.py).

StepLimitMiddleware counts the model calls of one run. On every call before the last it changes
nothing. On the last allowed call it offers the model no tools (`request.override(tools=[])`), so
the model writes a final answer, removes any tool call the model still made, logs
`step_limit_reached` and sets the `step_limit_reached` flag on the result:

    state = await agent.ainvoke({"messages": [...]}, config=...)
    if step_limit_reached(state):
        ...  # tell the user the answer was cut short

Both counters are LangGraph UntrackedValue channels: they live for one run and are never saved
with the thread, so every run starts again from zero. Gateway.agent() puts this middleware last
(innermost), so a caller's middleware cannot offer tools again on the last call.

LangChain's own ModelCallLimitMiddleware(exit_behavior="end") is not used: it ends the run with a
fixed message instead of a model answer.
Docs: https://docs.langchain.com/oss/python/langchain/middleware/custom
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import OmitFromSchema
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Command

from app.llm.core.steps import STEP_LIMIT_REACHED, StepLimit, log_reached

__all__ = ["StepLimitMiddleware", "recursion_limit", "step_limit_reached"]

_RUN_MODEL_CALLS = "run_model_calls"
_NOT_IN_INPUT = OmitFromSchema(input=True, output=False)


class StepLimitState(AgentState[Any]):
    run_model_calls: NotRequired[Annotated[int, UntrackedValue, _NOT_IN_INPUT]]
    step_limit_reached: NotRequired[Annotated[bool, UntrackedValue, _NOT_IN_INPUT]]


def step_limit_reached(result: Mapping[str, Any]) -> bool:
    """Whether an agent run ended on its forced final answer (the result of `ainvoke`)."""
    return bool(result.get(STEP_LIMIT_REACHED, False))


def recursion_limit(limit: StepLimit, middleware: Sequence[AgentMiddleware[Any, Any, Any]]) -> int:
    """LangGraph's own step limit, a backstop only. A run of N model calls takes 2N steps (N model
    steps, N - 1 tool steps and one more that LangGraph counts; measured, tests/test_llm_steps.py),
    plus N for each node a caller's middleware runs before or after every model call. Allow
    (2 + hooks) x N + 3, so that the limit never stops a run first and a graph that loops without
    calling the model still stops."""
    hooks = sum(_overrides(m, "before_model") + _overrides(m, "after_model") for m in middleware)
    return (2 + hooks) * limit.max_model_calls + 3


def _overrides(middleware: AgentMiddleware[Any, Any, Any], hook: str) -> int:
    cls = type(middleware)
    base = AgentMiddleware
    changed = getattr(cls, hook) is not getattr(base, hook)
    changed_async = getattr(cls, f"a{hook}") is not getattr(base, f"a{hook}")
    return int(changed or changed_async)


def _calls_made(request: ModelRequest[Any]) -> int:
    made = request.state.get(_RUN_MODEL_CALLS, 0)
    return made if isinstance(made, int) else 0


def _answer_without_tool_calls(response: ModelResponse[Any]) -> ModelResponse[Any]:
    """A model that ignores an empty tool list may still ask for a tool, which would send the run
    round the loop again. Remove the request, so the message is a plain answer. A structured
    answer (a ToolStrategy's own tool call) and its tool message are left as they are."""
    if response.structured_response is not None or any(
        isinstance(message, ToolMessage) for message in response.result
    ):
        return response
    result = [
        message.model_copy(update={"tool_calls": [], "invalid_tool_calls": []})
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls)
        else message
        for message in response.result
    ]
    return ModelResponse(result=result, structured_response=response.structured_response)


class StepLimitMiddleware(AgentMiddleware[StepLimitState, Any, Any]):
    state_schema = StepLimitState

    def __init__(self, limit: StepLimit) -> None:
        super().__init__()
        self.limit = limit

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ExtendedModelResponse[Any]:
        made = _calls_made(request)
        if self.limit.offers_tools(made):
            return self._after(handler(request), made, final=False)
        return self._after(handler(request.override(tools=[])), made, final=True)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ExtendedModelResponse[Any]:
        made = _calls_made(request)
        if self.limit.offers_tools(made):
            return self._after(await handler(request), made, final=False)
        return self._after(await handler(request.override(tools=[])), made, final=True)

    def _after(
        self, response: ModelResponse[Any], made: int, *, final: bool
    ) -> ExtendedModelResponse[Any]:
        update: dict[str, Any] = {_RUN_MODEL_CALLS: made + 1}
        if final:
            response = _answer_without_tool_calls(response)
            update[STEP_LIMIT_REACHED] = True
            log_reached(self.limit)
        return ExtendedModelResponse(model_response=response, command=Command(update=update))
