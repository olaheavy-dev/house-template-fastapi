"""LangGraph messages as context entries (#46, app/llm/langchain/context.py): roles, tool-call
groups, what is still waiting, and the usage LangChain stored on each answer."""

from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.llm.core.context import USAGE_KEY
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


def test_tool_arguments_shaped_like_a_document_are_not_an_attachment() -> None:
    # The budget looks for media in message content only (callbacks.py input_bound), so a call
    # whose arguments say "type": "document" is sent; the report must not refuse it either.
    call = {**CALL, "args": {"type": "document", "title": "House"}}
    got = entries([HumanMessage("File it."), AIMessage("", tool_calls=[call])])
    assert got[1].size > len("House")


def test_the_checked_bound_recorded_on_an_answer_is_never_sent_back() -> None:
    # The budget records the bound it checked in the answer's response_metadata (#46); the answer
    # goes back to the provider as history on the next call, its record must not.
    record = {USAGE_KEY: {"checked": 4_242}}
    history = [
        HumanMessage("What is 1 + 2?"),
        AIMessage("", tool_calls=[CALL], response_metadata=record),
        ToolMessage("3", tool_call_id="call_1"),
        AIMessage("3.", response_metadata=record),
        HumanMessage("And 3 + 3?"),
    ]
    models: list[Any] = [
        ChatAnthropic(model="claude-sonnet-5-5", api_key="unused"),
        ChatOpenAI(model="gpt-6-luna", api_key="unused"),
        ChatOpenAI(model="gpt-6-luna", api_key="unused", use_responses_api=True),
    ]
    for model in models:
        sent = json.dumps(model._get_request_payload(history), default=str)
        assert "And 3 + 3?" in sent
        assert USAGE_KEY not in sent
        assert "4242" not in sent
