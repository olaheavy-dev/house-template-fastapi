"""LangGraph messages as context entries (#46, app/llm/langchain/context.py): roles, tool-call
groups, what is still waiting, and the usage LangChain stored on each answer."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.llm.core.usage import Usage
from app.llm.langchain.context import entries, thread_messages

CALL = {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1", "type": "tool_call"}
USAGE = {"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500}


def test_roles_tool_calls_and_usage() -> None:
    got = entries(
        [
            SystemMessage("Be brief."),
            HumanMessage("What is 1 + 2?"),
            AIMessage("", tool_calls=[CALL], usage_metadata=USAGE),
            ToolMessage("3", tool_call_id="call_1"),
            AIMessage("3."),
        ]
    )
    assert [e.role for e in got] == ["system", "user", "assistant", "tool", "assistant"]
    one = frozenset({"call_1"})
    assert [e.links for e in got] == [frozenset(), frozenset(), one, one, frozenset()]
    assert not any(e.waiting for e in got)
    assert got[2].usage == Usage(input_tokens=1_000, output_tokens=500)
    assert got[4].usage is None
    assert all(e.size > 0 for e in got)


def test_a_tool_call_without_its_result_is_waiting() -> None:
    got = entries([HumanMessage("Add them."), AIMessage("", tool_calls=[CALL])])
    assert got[1].waiting


def test_a_missing_checkpoint_has_no_messages() -> None:
    assert thread_messages(None) == []
