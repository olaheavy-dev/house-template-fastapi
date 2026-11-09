"""Scripted chat models for the agent tests (#34). Named test_llm_*.py so that a project with the
database answer off drops it with the other model tests; it holds no tests of its own."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

__all__ = ["USAGE", "LoopingFake"]

USAGE: UsageMetadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


class LoopingFake(GenericFakeChatModel):
    """A model that asks for the `add` tool on every call that is offered any tool, and answers
    "final" on a call that is offered none: a confused model that never settles by itself.

    `offered` records, per call, the names of the tools it was offered (the order of the calls).
    `parallel` is how many `add` calls each turn asks for. `answer_on` makes call k (from 1)
    answer "final" even when tools are offered. `ignore_unbound` makes it ask for `add` even when
    no tool is offered, as a model that ignores an empty tool list would. When only a tool named
    `structured` is offered (a ToolStrategy's own tool), it calls that one with `structured_args`.
    """

    messages: Any = None  # unused: the script is this class's behaviour
    offered: list[list[str]] = Field(default_factory=list)
    parallel: int = 1
    answer_on: int | None = None
    ignore_unbound: bool = False
    # How a tool-calling message is shaped: "plain", "openai" (the calls also in additional_kwargs)
    # or "anthropic" (a tool_use block in list content), each with the text "Adding now".
    shape: str = "plain"
    structured: str | None = None
    structured_args: dict[str, Any] = Field(default_factory=dict)
    # Also ask for `add` in the same message as the structured tool call.
    stray_with_structured: bool = False
    # Call number (from 1) to the error that call raises instead of answering.
    raises: dict[int, BaseException] = Field(default_factory=dict)
    # Call number to something to run while that call is being made (after its reservation).
    hooks: dict[int, Callable[[], None]] = Field(default_factory=dict)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        names = [getattr(tool, "name", None) or tool["name"] for tool in tools]
        return self.bind(offered_tools=names, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        names: list[str] = kwargs.get("offered_tools", [])
        self.offered.append(names)
        number = len(self.offered)
        if number in self.hooks:
            self.hooks[number]()
        if number in self.raises:
            raise self.raises[number]
        message = AIMessage("final", usage_metadata=USAGE)
        if number != self.answer_on:
            if "add" in names or (self.ignore_unbound and not names):
                calls = [
                    {"name": "add", "args": {"a": 1, "b": 1}, "id": f"call-{number}-{i}"}
                    for i in range(self.parallel)
                ]
                message = self._tool_message(calls)
            elif self.structured in names:
                call = {"name": self.structured, "args": self.structured_args, "id": f"c-{number}"}
                calls = [call]
                if self.stray_with_structured:
                    calls.append({"name": "add", "args": {"a": 1, "b": 1}, "id": f"x-{number}"})
                message = AIMessage("", tool_calls=calls, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _tool_message(self, calls: list[dict[str, Any]]) -> AIMessage:
        if self.shape == "openai":
            raw = [
                {"id": c["id"], "type": "function", "function": {"name": "add", "arguments": "{}"}}
                for c in calls
            ]
            return AIMessage(
                "Adding now",
                tool_calls=calls,
                additional_kwargs={"tool_calls": raw},
                usage_metadata=USAGE,
            )
        if self.shape == "anthropic":
            blocks: list[str | dict[str, Any]] = [{"type": "text", "text": "Adding now"}]
            blocks += [
                {"type": "tool_use", "id": c["id"], "name": "add", "input": c["args"]}
                for c in calls
            ]
            return AIMessage(blocks, tool_calls=calls, usage_metadata=USAGE)
        return AIMessage("", tool_calls=calls, usage_metadata=USAGE)
