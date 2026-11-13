"""MAF messages as context entries (#46, app/llm/maf/context.py): roles, tool-call groups, what is
still waiting (a call without its result, an approval not yet given), and the usage the budget
middleware recorded on an answer."""

from __future__ import annotations

from agent_framework import Content, Message

from app.llm.core.context import USAGE_KEY
from app.llm.core.usage import Usage
from app.llm.maf.context import entries

CALL = Content.from_function_call(call_id="call_1", name="add", arguments={"a": 1, "b": 2})


def test_roles_tool_calls_and_usage() -> None:
    got = entries(
        [
            Message("system", ["Be brief."]),
            Message("user", ["What is 1 + 2?"]),
            Message("assistant", [CALL]),
            Message("tool", [Content.from_function_result(call_id="call_1", result="3")]),
            Message(
                "assistant",
                ["3."],
                additional_properties={USAGE_KEY: {"input_tokens": 1_000, "output_tokens": 500}},
            ),
        ]
    )
    assert [e.role for e in got] == ["system", "user", "assistant", "tool", "assistant"]
    one = frozenset({"call_1"})
    assert [e.links for e in got] == [frozenset(), frozenset(), one, one, frozenset()]
    assert not any(e.waiting for e in got)
    assert got[4].usage == Usage(input_tokens=1_000, output_tokens=500)
    assert got[2].usage is None


def test_a_tool_call_without_its_result_is_waiting() -> None:
    got = entries([Message("user", ["Add them."]), Message("assistant", [CALL])])
    assert got[1].waiting


def test_an_approval_request_waits_until_its_response() -> None:
    ask = Content.from_function_approval_request(id="ok_1", function_call=CALL)
    asked = [Message("user", ["Add them."]), Message("assistant", [ask])]
    assert entries(asked)[1].waiting
    answer = Content.from_function_approval_response(approved=True, id="ok_1", function_call=CALL)
    answered = entries([*asked, Message("user", [answer])])
    assert not answered[1].waiting
    # The request and its answer (a user message) are tied to the call, so they trim as one.
    assert answered[1].links == answered[2].links == frozenset({"call_1", "approval:ok_1"})
