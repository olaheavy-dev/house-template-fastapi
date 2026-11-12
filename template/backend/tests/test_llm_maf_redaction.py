"""Redaction over MAF's own message shapes (#78, #35): tool calls and their results, the
`instructions` option an agent sends its system prompt in, content the walkers do not know, and the
promise that the messages a caller holds are never changed. The values are synthetic."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from agent_framework import ChatResponse, Content, FunctionTool, Message, tool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core.redaction import RedactionFailed
from app.llm.maf.gateway import Gateway
from app.settings import Settings
from tests.harness.maf import HARNESS, ScriptedClient

pytestmark = pytest.mark.anyio

EMAIL = "priya.shah@example.com"
PHONE = "07700 900123"


@pytest.fixture
def gateway(settings: Settings, db_session: AsyncSession) -> Gateway:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    built = Gateway(
        settings.model_copy(update={"llm_model": "anthropic:claude-sonnet-5-5"}), lambda: sessions
    )
    built.meter.cap, built.meter.today = 1_000_000, lambda: date(2026, 1, 7)
    return built


def script(gateway: Gateway, *items: ChatResponse[Any]) -> ScriptedClient:
    HARNESS.install_scripted(
        gateway, list(items) or [ChatResponse(messages=[Message("assistant", ["ok"])])]
    )
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    return client


def calls_of(*contents: Content) -> list[Message]:
    return [Message("assistant", list(contents))]


async def test_a_tool_calls_arguments_are_redacted_as_a_string_and_as_a_mapping(
    gateway: Gateway,
) -> None:
    client = script(gateway)
    as_string = Content.from_function_call("c1", "send", arguments=json.dumps({"to": EMAIL}))
    as_mapping = Content.from_function_call("c2", "call", arguments={"number": PHONE})
    await gateway.client().get_response(calls_of(as_string, as_mapping))
    [sent] = client.seen[0]
    string_call, mapping_call = sent.contents
    assert json.loads(string_call.arguments) == {"to": "[EMAIL_1]"}  # type: ignore[arg-type]
    assert mapping_call.arguments == {"number": "[PHONE_1]"}
    # The caller's own objects are untouched: a stored thread keeps its originals.
    assert json.loads(as_string.arguments) == {"to": EMAIL}  # type: ignore[arg-type]
    assert as_mapping.arguments == {"number": PHONE}


async def test_a_tool_result_is_read_with_its_tools_name_in_front(gateway: Gateway) -> None:
    client = script(gateway)
    messages = [
        Message("assistant", [Content.from_function_call("c1", "get_date_of_birth", arguments={})]),
        Message("tool", [Content.from_function_result("c1", result="12/03/1980")]),
    ]
    await gateway.client().get_response(messages)
    result = client.seen[0][1].contents[0]
    # A bare date has no "date of birth" beside it; the tool's name supplies it.
    assert result.result == "[DOB_1]"
    assert messages[1].contents[0].result == "12/03/1980"


async def test_a_result_made_of_content_is_walked_too(gateway: Gateway) -> None:
    client = script(gateway)
    inner = Content.from_text(f"Mail {EMAIL}")
    messages = [
        Message("assistant", [Content.from_function_call("c1", "look", arguments={})]),
        Message("tool", [Content.from_function_result("c1", result=[inner])]),
    ]
    await gateway.client().get_response(messages)
    sent = client.seen[0][1].contents[0]
    # MAF keeps a result's text in `result` and its contents in `items`: both are cleaned.
    assert sent.result == "Mail [EMAIL_1]"
    assert sent.items is not None
    assert sent.items[0].text == "Mail [EMAIL_1]"
    assert inner.text == f"Mail {EMAIL}"


async def test_the_instructions_an_agent_sends_are_redacted_before_the_messages(
    gateway: Gateway,
) -> None:
    client = script(gateway)
    await gateway.client().get_response(
        [Message("user", [f"and {EMAIL}"])],
        options={"instructions": f"The user is at {EMAIL}."},
    )
    assert client.seen_options[0]["instructions"] == "The user is at [EMAIL_1]."
    # One numbering over the whole request: the same email is the same placeholder.
    assert client.seen[0][0].text == "and [EMAIL_1]"


async def test_the_originals_come_back_in_a_tool_call_the_model_makes(gateway: Gateway) -> None:
    reply = ChatResponse(
        messages=[
            Message(
                "assistant",
                [
                    Content.from_function_call(
                        "c1", "send", arguments=json.dumps({"to": "[EMAIL_1]", "n": "[PHONE_1]"})
                    ),
                    Content.from_function_call("c2", "send", arguments={"to": "[EMAIL_1]"}),
                ],
            )
        ]
    )
    HARNESS.install_plain(gateway, [reply])
    response = await gateway.client().get_response(
        [Message("user", [f"mail {EMAIL} or {PHONE}"])],
        options={"tools": [{"type": "function", "name": "send"}]},
    )
    first, second = response.messages[0].contents
    assert json.loads(first.arguments) == {"to": EMAIL, "n": PHONE}  # type: ignore[arg-type]
    assert second.arguments == {"to": EMAIL}


async def test_a_quote_in_an_original_does_not_break_the_json_of_a_restored_argument(
    gateway: Gateway,
) -> None:
    from app.llm.core.redaction import redaction_scope

    reply = ChatResponse(
        messages=[
            Message(
                "assistant",
                [Content.from_function_call("c1", "tell", arguments='{"who": "[NAME_1]"}')],
            )
        ]
    )
    HARNESS.install_plain(gateway, [reply])
    async with redaction_scope(known={"NAME": ['Priya "PJ" Shah']}):
        response = await gateway.client().get_response(
            [Message("user", ['Priya "PJ" Shah is here'])],
            options={"tools": [{"type": "function", "name": "tell"}]},
        )
    assert json.loads(response.messages[0].contents[0].arguments) == {"who": 'Priya "PJ" Shah'}  # type: ignore[arg-type]


async def test_signed_thinking_is_left_as_it_is_and_unsigned_thinking_is_cleaned(
    gateway: Gateway,
) -> None:
    client = script(gateway)
    signed = Content.from_text_reasoning(text=f"Maybe {EMAIL}?", protected_data="sig-abc")
    unsigned = Content.from_text_reasoning(text=f"Perhaps {EMAIL}?")
    await gateway.client().get_response(
        calls_of(signed, unsigned, Content.from_text(f"Mail {EMAIL}"))
    )
    sent = client.seen[0][0].contents
    assert sent[0].text == f"Maybe {EMAIL}?"  # a provider checks its signature
    assert sent[1].text == "Perhaps [EMAIL_1]?"  # sent as plain text by the Anthropic client
    assert sent[2].text == "Mail [EMAIL_1]"
    assert unsigned.text == f"Perhaps {EMAIL}?"


@pytest.mark.parametrize(
    "content",
    [
        Content.from_error(message=f"failed for {EMAIL}"),
        Content.from_oauth_consent_request(consent_link=f"https://x.example/?u={EMAIL}"),
    ],
)
async def test_content_the_walkers_do_not_know_is_refused_and_nothing_is_sent(
    gateway: Gateway, content: Content
) -> None:
    client = script(gateway)
    with capture_logs() as logs, pytest.raises(RedactionFailed) as raised:
        await gateway.client().get_response([Message("user", [content])])
    assert raised.value.status == 503
    assert client.seen == []
    assert EMAIL not in repr(logs)
    assert [log["event"] for log in logs] == ["llm_redaction_failed"]


async def test_an_author_name_that_holds_personal_data_is_refused(gateway: Gateway) -> None:
    client = script(gateway)
    with pytest.raises(RedactionFailed):
        await gateway.client().get_response([Message("user", ["hi"], author_name=EMAIL)])
    assert client.seen == []
    await gateway.client().get_response([Message("user", ["hi"], author_name="priya")])
    assert client.seen[0][0].author_name == "priya"


async def test_a_document_is_refused_by_the_budget_and_an_image_is_only_counted(
    gateway: Gateway,
) -> None:
    from app.llm.core.bounds import UnboundedMedia

    client = script(gateway)
    pdf = Content.from_uri("https://example.com/a.pdf", media_type="application/pdf")
    with pytest.raises(UnboundedMedia):
        await gateway.client().get_response([Message("user", [pdf])])
    assert client.seen == []
    png = Content.from_uri("https://example.com/a.png", media_type="image/png")
    await gateway.client().get_response([Message("user", [png])])
    assert len(client.seen) == 1


async def test_a_tool_object_of_an_unknown_type_is_refused_before_anything_is_sent(
    gateway: Gateway,
) -> None:
    from app.llm.core.bounds import UnpricedRequest

    # Redaction cannot read the text of a tool type it does not know, so it refuses first (fail
    # closed); with redaction off, the budget still refuses it as a tool it cannot price.
    client = script(gateway)
    with pytest.raises(RedactionFailed):
        await gateway.client().get_response(
            [Message("user", ["hi"])],
            options={"tools": [object()]},
        )
    HARNESS.set_redactor(gateway, None)
    with pytest.raises(UnpricedRequest):
        await gateway.client().get_response(
            [Message("user", ["hi"])],
            options={"tools": [object()]},
        )
    assert client.seen == []


def lead_tool(calls: list[str]) -> FunctionTool:
    def lookup_lead(who: str) -> str:
        """Look up a lead. Leads for priya.shah@example.com go to the office."""
        calls.append(who)
        return "found"

    return tool(lookup_lead)


async def test_a_tools_definition_is_sent_as_a_redacted_copy_and_the_callers_tool_runs(
    gateway: Gateway,
) -> None:
    calls: list[str] = []
    the_tool = lead_tool(calls)
    original = (the_tool.description, the_tool.parameters())
    asks = ChatResponse(
        messages=calls_of(
            Content.from_function_call("c1", "lookup_lead", arguments={"who": "[EMAIL_1]"})
        )
    )
    client = script(gateway, asks, ChatResponse(messages=[Message("assistant", ["done"])]))
    await gateway.client().get_response(
        [Message("user", [f"Is {EMAIL} a lead?"])], options={"tools": [the_tool]}
    )
    [sent] = client.seen_options[0]["tools"]
    assert sent is not the_tool
    assert sent.name == "lookup_lead"
    assert sent.description == "Look up a lead. Leads for [EMAIL_1] go to the office."
    # The caller's tool is unchanged, and it is the one MAF ran, with the original value.
    assert (the_tool.description, the_tool.parameters()) == original
    assert calls == [EMAIL]


async def test_a_tool_with_nothing_to_replace_is_sent_as_the_same_object(gateway: Gateway) -> None:
    def find_property(postcode: str) -> str:
        """Find a property by its postcode."""
        return postcode

    the_tool = tool(find_property)
    client = script(gateway)
    await gateway.client().get_response([Message("user", ["hi"])], options={"tools": [the_tool]})
    assert client.seen_options[0]["tools"][0] is the_tool


async def test_a_mapping_tool_is_redacted_too(gateway: Gateway) -> None:
    from app.llm.core.bounds import UnpricedRequest

    # MAF passes a mapping through as it is; the budget refuses a provider-run one, so whatever
    # redaction made of it, nothing is sent. A function-shaped mapping goes through, redacted.
    client = script(gateway)
    hosted = {"type": "web_search", "description": f"about {EMAIL}"}
    with pytest.raises(UnpricedRequest):
        await gateway.client().get_response([Message("user", ["hi"])], options={"tools": [hosted]})
    assert client.seen == []
    as_function: dict[str, Any] = {
        "type": "function",
        "function": {"name": "lookup", "description": f"for {EMAIL}", "parameters": {}},
    }
    await gateway.client().get_response([Message("user", ["hi"])], options={"tools": [as_function]})
    [sent] = client.seen_options[0]["tools"]
    assert sent["function"] == {"name": "lookup", "description": "for [EMAIL_1]", "parameters": {}}
    assert as_function["function"]["description"] == f"for {EMAIL}"


def test_the_function_tool_attribute_the_copy_sets_exists_in_the_pinned_core() -> None:
    # redaction._walk_tool sets a private attribute of agent-framework-core 1.25.0 on its copy:
    # a bump that renames it must fail here, not send the unredacted schema.
    import copy

    the_tool = lead_tool([])
    the_tool.parameters()
    assert hasattr(the_tool, "_cached_parameters")
    copied = copy.copy(the_tool)
    copied._cached_parameters = {"type": "object", "description": "[EMAIL_1]"}
    assert copied.parameters() == {"type": "object", "description": "[EMAIL_1]"}
    assert "[EMAIL_1]" not in str(the_tool.parameters())
