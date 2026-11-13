"""A LangGraph thread's messages as the core's context entries (#46, app/llm/core/context.py).

The usage comes from each answer's own `usage_metadata`, which LangChain sets from the provider's
report and LangGraph stores with the message. A tool call waits while any of its calls has no
ToolMessage yet (an agent paused for an approval, or a run that stopped part-way).

Sizes are the budget's own (app/llm/langchain/callbacks.py input_bound): each message's whole
serialised form, and the call's parameters with the tools bound, so the report's `bound` is never
below the one the budget checks against OpenAI's prompt cap. (With redaction on, the budget sizes
the redacted messages: a placeholder longer than the value it replaces adds a few bytes the report
can't see.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

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
                # The message as the budget serialises it, plus the ", " that separates it from the
                # next one in the budget's list.
                size=size_bound(message.model_dump()) + 2,
                links=frozenset(
                    [str(call.get("id")) for call in calls if call.get("id")]
                    + ([message.tool_call_id] if isinstance(message, ToolMessage) else [])
                ),
                waiting=any(call.get("id") not in answered for call in calls),
                usage=from_langchain(usage) if usage else None,
            )
        )
    return result


def fixed_size(model: BaseChatModel, system_prompt: str | None, tools: Sequence[Any]) -> int:
    """An upper bound of what an agent sends with every call besides the history, as the budget
    sizes it: the system message, and the parameters LangChain hands the budget for the call (the
    model's settings and, bound, the tool definitions). Computed as the budget's payload for no
    stored messages, so adding the entries' sizes gives at least its figure."""
    kwargs = model.bind_tools(list(tools)).kwargs if tools else {}  # type: ignore[attr-defined]
    # What LangChain itself passes to callbacks as `invocation_params` (BaseChatModel.ainvoke).
    params = model._get_invocation_params(**kwargs)
    system = [SystemMessage(system_prompt).model_dump()] if system_prompt else []
    return size_bound([system, params])


def thread_messages(checkpoint: Any) -> list[BaseMessage]:
    """The messages a checkpoint holds under the `messages` key (create_agent's state, and any
    graph built on MessagesState), or none."""
    if checkpoint is None:
        return []
    messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
    return [m for m in messages if isinstance(m, BaseMessage)]
