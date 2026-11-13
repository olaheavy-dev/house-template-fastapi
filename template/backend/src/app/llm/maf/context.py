"""A MAF session's messages as the core's context entries (#46, app/llm/core/context.py).

MAF keeps no call's usage with the messages it stores, so the budget middleware records it: after
settling a call, it writes the call's input and output tokens under USAGE_KEY in the answer
message's `additional_properties` (founder decision, 13 Nov 2026). Those properties are saved with
the session and never sent to the provider (checked on agent-framework-core 1.25.0, the OpenAI
Responses and Anthropic clients). A streamed run builds its answer from the
updates, which carry no message properties: there the middleware sends the record as one last
update with no content, MAF merges it into the run's response, and UsageHistoryProvider (the
history provider the gateway gives an agent) moves it onto the answer before storing it. A message
without the record, such as one stored before this existed or by another history provider, is
estimated.

The messages are those of the agent's InMemoryHistoryProvider (the gateway adds one when no
context provider is given), from `session.state[<its source_id>]["messages"]`. A tool call waits
while one of its calls has no result yet, and an approval request waits until its response is in
the history.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_framework import AgentSession, InMemoryHistoryProvider, Message, SessionContext

from app.llm.core.context import USAGE_KEY, Entry, Role, size_bound, usage_from_record
from app.llm.maf.middleware import request_view

__all__ = ["UsageHistoryProvider", "entries", "fixed_size", "history"]

_ROLES: dict[str, Role] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


class UsageHistoryProvider(InMemoryHistoryProvider):
    """MAF's InMemoryHistoryProvider (same `source_id`, so it reads sessions stored before it),
    which first moves the last call's usage record from a streamed run's response onto its answer
    (module notes)."""

    async def after_run(
        self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        response = context.response
        if response is not None and response.messages:
            record = (response.additional_properties or {}).get(USAGE_KEY)
            answer = response.messages[-1]
            if answer.additional_properties is None:
                answer.additional_properties = {}
            if record is not None and USAGE_KEY not in answer.additional_properties:
                answer.additional_properties[USAGE_KEY] = record
        await super().after_run(agent=agent, session=session, context=context, state=state)


def _approval_links(content: Any) -> set[str]:
    """An approval request or answer's links: its own id, and its function call's id."""
    links = {f"approval:{content.id}"} if content.id else set()
    call = content.function_call
    if call is not None and call.call_id:
        links.add(str(call.call_id))
    return links


def _links(message: Message) -> frozenset[str]:
    links: set[str] = set()
    for content in message.contents:
        if content.type in ("function_call", "function_result") and content.call_id:
            links.add(str(content.call_id))
        elif content.type in ("function_approval_request", "function_approval_response"):
            links |= _approval_links(content)
    return frozenset(links)


def _ids(messages: Sequence[Message], kind: str) -> set[str]:
    return {
        str(content.call_id if kind == "function_result" else content.id)
        for message in messages
        for content in message.contents
        if content.type == kind
    }


def entries(messages: Sequence[Message]) -> list[Entry]:
    """One entry per message, in order."""
    results = _ids(messages, "function_result")
    approved = _ids(messages, "function_approval_response")
    found = []
    for message in messages:
        payload, blocks = request_view([message], {})
        calls = [str(c.call_id) for c in message.contents if c.type == "function_call"]
        asks = [str(c.id) for c in message.contents if c.type == "function_approval_request"]
        found.append(
            Entry(
                role=_ROLES.get(str(message.role), "assistant"),
                size=size_bound([payload[0], blocks]),
                links=_links(message),
                waiting=any(call not in results for call in calls)
                or any(ask not in approved for ask in asks),
                usage=usage_from_record((message.additional_properties or {}).get(USAGE_KEY)),
            )
        )
    return found


def fixed_size(agent: Any) -> int:
    """An upper bound of what the agent sends with every call besides the history: its
    instructions and tool definitions (its `default_options`), sized as the budget sizes them."""
    options = dict(getattr(agent, "default_options", None) or {})
    payload, _ = request_view([], options)
    return size_bound(payload)


def history(agent: Any, session: AgentSession) -> list[Message]:
    """The messages the agent's InMemoryHistoryProvider keeps in `session`, oldest first."""
    messages: list[Message] = []
    for provider in getattr(agent, "context_providers", None) or ():
        if isinstance(provider, InMemoryHistoryProvider):
            state = session.state.get(provider.source_id) or {}
            messages += [m for m in state.get("messages", []) if isinstance(m, Message)]
    return messages
