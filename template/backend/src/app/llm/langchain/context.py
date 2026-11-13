"""A LangGraph thread's messages as the core's context entries (#46, app/llm/core/context.py).

The usage comes from each answer's own `usage_metadata`, which LangChain sets from the provider's
report and LangGraph stores with the message. A tool call waits while any of its calls has no
ToolMessage yet (an agent paused for an approval, or a run that stopped part-way).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.llm.core.context import Entry, Role, size_bound
from app.llm.langchain.usage import from_langchain

__all__ = ["entries", "fixed_size", "thread_messages"]

_ROLES: dict[str, Role] = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def entries(messages: Sequence[BaseMessage]) -> list[Entry]:
    """One entry per message, in order."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    result = []
    for message in messages:
        calls = message.tool_calls if isinstance(message, AIMessage) else []
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        result.append(
            Entry(
                role=_ROLES.get(message.type, "assistant"),
                size=size_bound([message.content, calls]),
                links=frozenset(
                    [str(call.get("id")) for call in calls if call.get("id")]
                    + ([message.tool_call_id] if isinstance(message, ToolMessage) else [])
                ),
                waiting=any(call.get("id") not in answered for call in calls),
                usage=from_langchain(usage) if usage else None,
            )
        )
    return result


def fixed_size(system_prompt: str | None, tools: Sequence[Any]) -> int:
    """An upper bound of what an agent sends with every call besides the history: its system
    prompt and tool definitions (as OpenAI-style tool schemas)."""
    specs = [convert_to_openai_tool(tool) for tool in tools]
    return size_bound([system_prompt or "", specs])


def thread_messages(checkpoint: Any) -> list[BaseMessage]:
    """The messages a checkpoint holds under the `messages` key (create_agent's state, and any
    graph built on MessagesState), or none."""
    if checkpoint is None:
        return []
    messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
    return [m for m in messages if isinstance(m, BaseMessage)]
