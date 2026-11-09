"""The agent step limit for LangChain's create_agent (the rule is app/llm/core/steps.py).

StepLimitMiddleware counts the model calls of one run. On every call before the last it changes
nothing. On the last allowed call it offers the model no tools (final_request: `tools=[]`, or
tool_choice none for Anthropic with a ProviderStrategy), so the model writes a final answer,
removes any tool call the model still made, logs
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

from langchain.agents.factory import _supports_provider_strategy
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import OmitFromSchema
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Command

from app.llm.core.steps import STEP_LIMIT_REACHED, StepLimit, log_reached

__all__ = ["StepLimitMiddleware", "final_request", "recursion_limit", "step_limit_reached"]

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
    plus N for each node a caller's middleware runs before or after every model call (h of them),
    plus one for each node it runs once before or after the whole run (a of them). Allow
    (2 + h) x N + a + 3, so that the limit never stops a run first and a graph that loops
    without calling the model still stops."""
    hooks = sum(_overrides(m, "before_model") + _overrides(m, "after_model") for m in middleware)
    run_hooks = sum(
        _overrides(m, "before_agent") + _overrides(m, "after_agent") for m in middleware
    )
    return (2 + hooks) * limit.max_model_calls + run_hooks + 3


def _overrides(middleware: AgentMiddleware[Any, Any, Any], hook: str) -> int:
    cls = type(middleware)
    base = AgentMiddleware
    changed = getattr(cls, hook) is not getattr(base, hook)
    changed_async = getattr(cls, f"a{hook}") is not getattr(base, f"a{hook}")
    return int(changed or changed_async)


def final_request(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request for the last call: the model may not call a tool.

    Offering no tools (`tools=[]`) was run against the real APIs on 9 Nov 2026 (OpenAI gpt-6-luna,
    Anthropic claude-haiku-5-5), in three cases each, a plain agent, a ToolStrategy and a
    ProviderStrategy, with a second run on the saved thread: it was answered everywhere except
    Anthropic with a ProviderStrategy, where the request got a 400 whose message was about a
    thinking block's signature (the cause was not isolated). There the tools stay bound and
    tool_choice {"type": "none"} (Anthropic's documented value) forbids calling them; that was
    answered. AutoStrategy and other providers were not run against real APIs.

    create_agent passes only model_settings to bind_tools when it uses a ProviderStrategy, and
    passes tool_choice itself otherwise, so the choice goes in model_settings and only on that path.
    An AutoStrategy (or a bare schema) is resolved here as create_agent resolves it."""
    if _anthropic(request.model) and _uses_provider_strategy(request):
        settings = {**request.model_settings, "tool_choice": {"type": "none"}}
        return request.override(model_settings=settings)
    return request.override(tools=[])


def _anthropic(model: Any) -> bool:
    # Not an isinstance check: the provider SDK packages are imported only by the gateway
    # (lint-imports). ChatAnthropic's _llm_type is "anthropic-chat".
    return getattr(model, "_llm_type", None) == "anthropic-chat"


def _uses_provider_strategy(request: ModelRequest[Any]) -> bool:
    """Whether create_agent will bind the model for a ProviderStrategy (langchain.agents.factory,
    _get_bound_model): an explicit one, or an AutoStrategy or bare schema on a model whose profile
    says it has native structured output."""
    response_format = request.response_format
    if response_format is None or isinstance(response_format, ToolStrategy):
        return False
    if isinstance(response_format, ProviderStrategy):
        return True
    # AutoStrategy, or a raw schema create_agent turns into one.
    return bool(_supports_provider_strategy(request.model, tools=request.tools))


def _calls_made(request: ModelRequest[Any]) -> int:
    made = request.state.get(_RUN_MODEL_CALLS, 0)
    return made if isinstance(made, int) else 0


def _plain(message: AIMessage) -> AIMessage:
    """The message with only its text. A copy that merely emptied tool_calls would keep the
    provider's own copies (OpenAI's additional_kwargs["tool_calls"], Anthropic's tool_use blocks),
    which the provider reads back on the next call."""
    return AIMessage(
        content=message.text,
        id=message.id,
        name=message.name,
        usage_metadata=message.usage_metadata,
        response_metadata=message.response_metadata,
    )


def _answer_without_tool_calls(response: ModelResponse[Any]) -> ModelResponse[Any]:
    """A model that ignores an empty tool list may still ask for a tool, which would send the run
    round the loop again. Remove the request, so the message is a plain answer. A valid structured
    answer (a ToolStrategy's own tool call and its tool message) is left as it is. A structured
    answer that failed validation is not: create_agent would answer it with an error message and
    ask the model again, past the limit, so the error message is dropped and the answer is the
    text alone, and the run ends whatever caller hooks run after the model."""
    if response.structured_response is not None:
        return response
    result = [
        _plain(message)
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls)
        else message
        for message in response.result
        if not isinstance(message, ToolMessage)
    ]
    return ModelResponse(result=result, structured_response=None)


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
        self._check(made)
        if self.limit.offers_tools(made):
            return self._after(handler(request), made, final=False)
        return self._after(handler(final_request(request)), made, final=True)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ExtendedModelResponse[Any]:
        made = _calls_made(request)
        self._check(made)
        if self.limit.offers_tools(made):
            return self._after(await handler(request), made, final=False)
        return self._after(await handler(final_request(request)), made, final=True)

    def _check(self, made: int) -> None:
        # The last call ends the run (see _after), so this is a bug if it ever happens.
        if made >= self.limit.max_model_calls:
            raise RuntimeError(
                f"The model-call limit of {self.limit.max_model_calls} was already used up"
            )

    def _after(
        self, response: ModelResponse[Any], made: int, *, final: bool
    ) -> ExtendedModelResponse[Any]:
        update: dict[str, Any] = {_RUN_MODEL_CALLS: made + 1}
        if final:
            response = _answer_without_tool_calls(response)
            update[STEP_LIMIT_REACHED] = True
            # End the run here whatever the answer: a structured answer that fails validation
            # would otherwise send the graph back to the model for call N + 1. Such a run has no
            # structured_response.
            update["jump_to"] = "end"
            log_reached(self.limit)
        return ExtendedModelResponse(model_response=response, command=Command(update=update))
