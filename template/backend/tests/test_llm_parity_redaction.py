"""What every model framework must do with the text it sends (#78): remove personal data first,
before the budget reserves anything, and put the originals back in the answer. Two parts: chat
calls (the end of this file's first half) and the text sent for embeddings, which is cleaned one
way, each text on its own. The database is real; the chat provider and OpenAI's embeddings endpoint
are faked (tests/harness/), so each test runs once per framework the project chose and imports
none.

The values are synthetic: the email and phone are from reserved or never-allocated ranges, the name
is made up. A placeholder is [KIND_n]; "[EMAIL_1]" is how every email below is sent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Annotated, Any

import anyio
import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall
from app.llm.core.redaction import RedactionFailed, Redactor, redaction_scope
from app.settings import Settings
from tests.evals.redaction_corpus import MUST_NOT_CHANGE
from tests.harness import FakeEmbeddingsApi, Harness, Reply, user

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 3)
CHAT_DAY = date(2026, 1, 4)
CHAT_MODEL = "anthropic:claude-sonnet-5-5"
IMAGE = {"type": "image", "url": "https://example.com/floorplan.png"}
EMAIL = "priya.shah@example.com"
PHONE = "07700 900123"  # Ofcom's drama range, never allocated
NAME = "Priya Shah"


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def make(
    framework: Harness, settings: Settings, sessions: async_sessionmaker[AsyncSession]
) -> Callable[..., tuple[Any, FakeEmbeddingsApi]]:
    """A gateway (redaction on unless asked) whose embeddings go to a fake of OpenAI's endpoint."""

    def build(**options: Any) -> tuple[Any, FakeEmbeddingsApi]:
        gateway = framework.gateway(settings.model_copy(update=options), lambda: sessions)
        fake = FakeEmbeddingsApi()
        framework.install_embeddings(gateway, fake)
        meter = framework.embedding_meter(gateway)
        meter.cap, meter.today = 1_000_000, lambda: DAY
        return gateway, fake

    return build


# --- chat calls --------------------------------------------------------------------------------


@pytest.fixture
def chat(
    framework: Harness, settings: Settings, sessions: async_sessionmaker[AsyncSession]
) -> Callable[..., Any]:
    """A gateway (redaction on unless asked) priced for CHAT_MODEL, on its own day of the ledger."""

    def build(*script: Reply, **options: Any) -> Any:
        gateway = framework.gateway(
            settings.model_copy(update={"llm_model": CHAT_MODEL, **options}), lambda: sessions
        )
        meter = framework.meter(gateway)
        meter.cap, meter.today = 1_000_000, lambda: CHAT_DAY
        framework.install_scripted(gateway, list(script) or [Reply("ok")] * 4)
        return gateway

    return build


async def chat_rows(session: AsyncSession) -> list[LlmCall]:
    rows = await session.scalars(
        select(LlmCall).where(LlmCall.day == CHAT_DAY).order_by(LlmCall.created_at)
    )
    return list(rows)


async def test_the_chat_provider_gets_placeholders_by_default_never_the_values(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat()
    await framework.ask(gateway, [user(f"Write to {EMAIL} or call {PHONE}")])
    assert framework.last_request_messages(gateway) == [
        user("Write to [EMAIL_1] or call [PHONE_1]")
    ]


async def test_the_originals_are_put_back_in_the_answer_and_an_invented_placeholder_stays(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(Reply("Done: I wrote to [EMAIL_1] and [EMAIL_7]."))
    with capture_logs() as logs:
        answer = await framework.ask(gateway, [user(f"Please email {EMAIL}")])
    # The round trip (test 14): [EMAIL_1] is the email again; [EMAIL_7] was never sent, so it is
    # left as the model wrote it and counted once as unrestored (a count, never a value).
    assert answer == f"Done: I wrote to {EMAIL} and [EMAIL_7]."
    assert [log["count"] for log in logs if log["event"] == "llm_unrestored_placeholders"] == [1]


async def test_the_values_the_app_names_are_removed_from_a_chat_call_too(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(Reply("Hello [NAME_1]"))
    async with redaction_scope(known={"NAME": [NAME]}, allow=["020 7946 0321"]):
        answer = await framework.ask(gateway, [user(f"{NAME} rang 020 7946 0321")])
    assert framework.last_request_messages(gateway) == [user("[NAME_1] rang 020 7946 0321")]
    assert answer == f"Hello {NAME}"


async def test_text_about_a_property_is_sent_to_the_model_as_it_is(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat()
    await framework.ask(gateway, [user(text) for text in MUST_NOT_CHANGE])
    assert framework.last_request_messages(gateway) == [user(text) for text in MUST_NOT_CHANGE]


async def test_a_chat_call_that_cannot_be_cleaned_is_a_503_and_nothing_is_sent_or_written(
    framework: Harness, chat: Callable[..., Any], db_session: AsyncSession
) -> None:
    class Breaks:
        kind = "EMAIL"

        def find(self, text: str) -> Any:
            raise RuntimeError(f"cannot read {text}")  # an error whose message holds the text

    gateway = chat()
    framework.set_redactor(gateway, Redactor([Breaks()]))
    with capture_logs() as logs, pytest.raises(RedactionFailed) as raised:
        await framework.ask(gateway, [user(f"fine, then {EMAIL}")])
    # Test 21: a 503, the provider was never asked, no ledger row, and only the error's type name
    # was logged.
    assert raised.value.status == 503
    assert framework.scripted_left(gateway) == 4
    assert await chat_rows(db_session) == []
    assert EMAIL not in str(raised.value) + repr(logs)
    assert [log["error_type"] for log in logs if log["event"] == "llm_redaction_failed"] == [
        "RuntimeError"
    ]


async def test_a_chat_calls_row_counts_the_values_replaced_and_is_null_only_when_redaction_is_off(
    framework: Harness, chat: Callable[..., Any], db_session: AsyncSession
) -> None:
    gateway = chat()
    await framework.ask(gateway, [user(f"{EMAIL}, {PHONE}"), user(f"and again {EMAIL}")])  # 3
    await framework.ask(gateway, [user("a freehold")])  # redaction ran, found nothing
    off = chat(llm_redaction="off")
    await framework.ask(off, [user(EMAIL)])  # redaction off
    # Rows of one test share created_at (now() is the transaction's start), so compare as a set.
    assert {row.redacted_count for row in await chat_rows(db_session)} == {3, 0, None}


async def test_with_redaction_off_the_messages_are_sent_exactly_as_they_are(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(llm_redaction="off")
    await framework.ask(gateway, [user(f"Write to {EMAIL}")])
    assert framework.last_request_messages(gateway) == [user(f"Write to {EMAIL}")]


async def test_a_call_is_reserved_for_the_text_that_is_sent_not_the_text_given(
    framework: Harness, chat: Callable[..., Any], db_session: AsyncSession
) -> None:
    # Test 28. The email is 22 bytes and its placeholder 9: the redacted request is the bytes of
    # the second call, so the ledger reserves the same for both, and less than with redaction off.
    gateway = chat()
    await framework.ask(gateway, [user(f"Write to {EMAIL} please")])
    await framework.ask(gateway, [user("Write to [EMAIL_1] please")])
    off = chat(llm_redaction="off")
    await framework.ask(off, [user(f"Write to {EMAIL} please")])
    # Each row is told apart by its count (1, 0, NULL when off), not by created_at: the rows of one
    # test share it (now() is the transaction's start), so their order is not defined.
    by_count = {r.redacted_count: r.reserved_micro_usd for r in await chat_rows(db_session)}
    redacted, as_written, unredacted = by_count[1], by_count[0], by_count[None]
    assert redacted == as_written
    assert unredacted > redacted


async def test_an_image_is_counted_as_unscanned_and_the_text_beside_it_is_still_cleaned(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat()
    with capture_logs() as logs:
        await framework.ask(gateway, [user([IMAGE, {"type": "text", "text": f"Mail {EMAIL}"}])])
    # Test 30: the image is sent as it is and counted (never read); the text beside it is cleaned.
    assert [log["count"] for log in logs if log["event"] == "llm_unscanned_images"] == [1]
    [sent] = framework.last_request_messages(gateway) or []
    assert "Mail [EMAIL_1]" in repr(sent["content"])
    assert EMAIL not in repr(sent["content"])


async def test_the_logs_of_a_chat_call_hold_no_value(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(Reply("[NAME_1] [EMAIL_1] [PHONE_1] [NAME_2]"))
    with capture_logs() as logs:
        async with redaction_scope(known={"NAME": [NAME]}, allow=[f"{NAME} Jr"]):
            await framework.ask(gateway, [user(f"{NAME} Jr, {NAME}, {EMAIL}, {PHONE}")])
    text = repr(logs)
    for value in (EMAIL, PHONE, NAME):
        assert value not in text


async def test_a_streamed_answer_gets_the_originals_back_even_when_a_placeholder_is_split(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    # Tests 15 and 16: the placeholder arrives in three pieces; what the consumer reads joins to
    # the restored answer, and the provider was sent the placeholder.
    gateway = chat(Reply("Mail [EMAIL_1] now", pieces=("Mail [EM", "AIL", "_1] now")))
    pieces = await framework.stream(gateway, [user(f"Write to {EMAIL}")])
    assert "".join(pieces) == f"Mail {EMAIL} now"
    assert not any("[EM" in piece for piece in pieces)
    assert framework.last_request_messages(gateway) == [user("Write to [EMAIL_1]")]


async def test_a_streamed_answer_counts_an_invented_placeholder_once(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(Reply("[EMAIL_1] and [EMAIL_9]", pieces=("[EMAIL_1] and [EM", "AIL_9]")))
    with capture_logs() as logs:
        pieces = await framework.stream(gateway, [user(f"Please email {EMAIL}")])
    assert "".join(pieces) == f"{EMAIL} and [EMAIL_9]"
    assert [log["count"] for log in logs if log["event"] == "llm_unrestored_placeholders"] == [1]


async def test_a_streamed_calls_row_counts_the_values_replaced(
    framework: Harness, chat: Callable[..., Any], db_session: AsyncSession
) -> None:
    gateway = chat(Reply("ok", pieces=("o", "k")))
    await framework.stream(gateway, [user(f"{EMAIL}, {PHONE}")])
    [row] = await chat_rows(db_session)
    assert (row.status, row.redacted_count) == ("settled", 2)


def lookup_lead(
    who: Annotated[str, Field(description="Their address, as priya.shah@example.com is")],
) -> str:
    """Look up a lead. Leads for priya.shah@example.com go to the office."""
    return who


def find_property(postcode: str) -> str:
    """Find a property by its postcode."""
    return postcode


class Lead(BaseModel):
    """A lead. Never write to priya.shah@example.com."""

    email: str = Field(description="The lead's address", examples=["priya.shah@example.com"])


async def test_a_tools_definition_is_sent_with_placeholders_and_its_name_as_it_is(
    framework: Harness, chat: Callable[..., Any], db_session: AsyncSession
) -> None:
    gateway = chat(Reply("Ask [EMAIL_1]"))
    answer = await framework.ask_with_tool(gateway, [user(f"Is {EMAIL} a lead?")], lookup_lead)
    [tool] = framework.last_request_tools(gateway)
    # The description is text the provider reads; the name is an identifier it checks. The same
    # email in the definition and in the message is one value with one placeholder, numbered in
    # the order the provider reads them (definitions first).
    assert tool["name"] == "lookup_lead"
    assert tool["description"] == "Look up a lead. Leads for [EMAIL_1] go to the office."
    assert tool["parameters"]["properties"]["who"]["description"] == (
        "Their address, as [EMAIL_1] is"
    )
    assert EMAIL not in repr(tool)
    assert framework.last_request_messages(gateway) == [user("Is [EMAIL_1] a lead?")]
    assert answer == f"Ask {EMAIL}"
    [row] = await chat_rows(db_session)
    assert row.redacted_count == 3


async def test_an_output_schema_is_sent_with_placeholders_and_parsed_into_the_callers_class(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat(Reply('{"email": "[EMAIL_1]"}'))
    parsed = await framework.ask_structured(gateway, [user("Extract the lead.")], Lead)
    schema = framework.last_request_schema(gateway)
    assert schema is not None
    assert EMAIL not in repr(schema)
    assert schema["description"] == "A lead. Never write to [EMAIL_1]."
    assert schema["properties"]["email"]["examples"] == ["[EMAIL_1]"]
    assert schema["title"] == "Lead"
    # The answer is the caller's own class, with the original back in its field.
    assert type(parsed) is Lead
    assert parsed.email == EMAIL


async def test_a_structured_answer_that_does_not_fit_its_schema_is_a_502_settled_at_its_cost(
    framework: Harness,
    chat: Callable[..., Any],
    app: FastAPI,
    client: httpx.AsyncClient,
    db_session: AsyncSession,
) -> None:
    # The founder's decision on #35 (11 Nov 2026): the same 502 under every framework, the call
    # settled at what it cost (the provider did the work), and nothing of the answer in the
    # response (it is the model's text, and can hold the user's data).
    gateway = chat(Reply('{"email": 5, "note": "priya.shah@example.com"}'))

    @app.get("/lead")
    async def lead() -> None:
        await framework.ask_structured(gateway, [user("Extract the lead.")], Lead)

    with capture_logs() as logs:
        response = await client.get("/lead")
    assert response.status_code == 502
    assert response.headers["content-type"] == "application/problem+json"
    assert (
        response.json()["detail"] == "The language model's answer didn't match the expected format."
    )
    assert EMAIL not in response.text
    assert EMAIL not in repr(logs)
    assert [log["event"] for log in logs if "schema" in log["event"]] == ["llm_schema_mismatch"]
    [row] = await chat_rows(db_session)
    assert row.status == "settled"
    assert row.cost_micro_usd is not None
    assert 0 < row.cost_micro_usd <= row.reserved_micro_usd


async def test_definitions_with_nothing_to_replace_or_with_redaction_off_are_sent_as_given(
    framework: Harness, chat: Callable[..., Any]
) -> None:
    gateway = chat()
    await framework.ask_with_tool(gateway, [user("hi")], find_property)
    [tool] = framework.last_request_tools(gateway)
    assert tool["description"] == "Find a property by its postcode."
    off = chat(Reply("ok"), Reply('{"email": "x"}'), llm_redaction="off")
    await framework.ask_with_tool(off, [user("hi")], lookup_lead)
    [tool] = framework.last_request_tools(off)
    assert EMAIL in tool["description"]
    await framework.ask_structured(off, [user("hi")], Lead)
    assert framework.last_request_schema(off) == Lead.model_json_schema()


async def counts(session: AsyncSession) -> list[int | None]:
    rows = await session.scalars(select(LlmCall.redacted_count).where(LlmCall.day == DAY))
    return sorted(rows, key=lambda count: -1 if count is None else count)


async def test_the_provider_gets_placeholders_never_the_values(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    await framework.embed_documents(gateway, [f"Write to {EMAIL} or call {PHONE}", "a freehold"])
    assert fake.requests == [["Write to [EMAIL_1] or call [PHONE_1]", "a freehold"]]


async def test_a_query_is_cleaned_exactly_like_a_document(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    await framework.embed_documents(gateway, [f"who is {EMAIL}"])
    await framework.embed_query(gateway, f"who is {EMAIL}")
    # The same text, the same bytes: a query finds the document it was made from.
    assert fake.requests == [["who is [EMAIL_1]"], ["who is [EMAIL_1]"]]


async def test_the_values_the_app_names_are_removed_and_the_allowed_ones_stay(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    async with redaction_scope(known={"NAME": [NAME]}, allow=["020 7946 0321"]):
        await framework.embed_documents(gateway, [f"{NAME} asked the agent on 020 7946 0321"])
    assert fake.requests == [["[NAME_1] asked the agent on 020 7946 0321"]]


async def test_the_scope_of_the_call_decides_what_is_removed(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    text = f"{NAME} wrote"
    await framework.embed_documents(gateway, [text])  # no scope: a name is not guessed
    async with redaction_scope(known={"NAME": [NAME]}):
        await framework.embed_documents(gateway, [text])
    assert fake.requests == [[text], ["[NAME_1] wrote"]]


async def test_text_about_a_property_is_sent_as_it_is(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    await framework.embed_documents(gateway, list(MUST_NOT_CHANGE))
    assert fake.requests == [list(MUST_NOT_CHANGE)]


async def test_a_text_is_sent_the_same_whatever_its_neighbours(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make()
    text = f"{EMAIL} and {PHONE}"
    await framework.embed_documents(gateway, [text])
    await framework.embed_documents(gateway, ["b@example.com", text, "c@example.com"])
    # Numbered inside each text: [EMAIL_1] is this email whichever texts surround it.
    assert fake.requests == [
        ["[EMAIL_1] and [PHONE_1]"],
        ["[EMAIL_1]", "[EMAIL_1] and [PHONE_1]", "[EMAIL_1]"],
    ]


async def test_the_row_counts_the_values_replaced_and_is_null_only_when_redaction_is_off(
    framework: Harness,
    make: Callable[..., tuple[Any, FakeEmbeddingsApi]],
    db_session: AsyncSession,
) -> None:
    gateway, _ = make()
    await framework.embed_documents(gateway, [f"{EMAIL}, {PHONE}", f"x {EMAIL}"])  # 3 values
    await framework.embed_documents(gateway, ["a freehold"])  # redaction ran, found nothing
    off, _ = make(llm_redaction="off")
    await framework.embed_documents(off, [EMAIL])  # redaction off
    assert await counts(db_session) == [None, 0, 3]


async def test_a_text_that_cannot_be_cleaned_is_a_503_and_nothing_is_sent_or_written(
    framework: Harness,
    make: Callable[..., tuple[Any, FakeEmbeddingsApi]],
    db_session: AsyncSession,
) -> None:
    class Breaks:
        kind = "EMAIL"

        def find(self, text: str) -> Any:
            raise RuntimeError(f"cannot read {text}")  # an error whose message holds the text

    gateway, fake = make()
    framework.set_redactor(gateway, Redactor([Breaks()]))
    with capture_logs() as logs, pytest.raises(RedactionFailed) as raised:
        await framework.embed_documents(gateway, ["fine", EMAIL])
    assert raised.value.status == 503
    assert fake.requests == []
    assert await counts(db_session) == []
    assert EMAIL not in str(raised.value) + repr(logs)


async def test_with_redaction_off_the_texts_are_sent_as_they_are(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, fake = make(llm_redaction="off")
    await framework.embed_documents(gateway, [f"Write to {EMAIL}"])
    await framework.embed_query(gateway, f"Write to {EMAIL}")
    assert fake.requests == [[f"Write to {EMAIL}"], [f"Write to {EMAIL}"]]


class MeetingApi(FakeEmbeddingsApi):
    """Answers only once both calls are inside it, so they are in flight together."""

    def __init__(self) -> None:
        super().__init__()
        self.arrived = 0
        self.both_in = anyio.Event()

    async def create(self, *, model: str, input: list[str], **kwargs: Any) -> Any:
        self.arrived += 1
        if self.arrived == 2:
            self.both_in.set()
        await self.both_in.wait()
        return await super().create(model=model, input=input, **kwargs)


async def test_calls_running_at_the_same_time_keep_their_own_scopes(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, _ = make()
    api = MeetingApi()
    framework.install_embeddings(gateway, api)
    # The test's one database connection runs one statement at a time: the ledger calls take turns
    # (the sends do not).
    meter = framework.embedding_meter(gateway)
    turn = anyio.Lock()
    reserve, settle = meter.reserve, meter.settle

    async def reserve_in_turn(*args: Any, **kwargs: Any) -> Any:
        async with turn:
            return await reserve(*args, **kwargs)

    async def settle_in_turn(*args: Any, **kwargs: Any) -> Any:
        async with turn:
            return await settle(*args, **kwargs)

    meter.reserve, meter.settle = reserve_in_turn, settle_in_turn  # type: ignore[method-assign]

    async def run(person: str) -> None:
        async with redaction_scope(known={"NAME": [person]}):
            await framework.embed_documents(gateway, ["Alice Example met Bob Sample"])

    async with anyio.create_task_group() as group:
        group.start_soon(run, "Alice Example")
        group.start_soon(run, "Bob Sample")
    # Each call removed only its own person, though both were in flight at once.
    assert sorted(request[0] for request in api.requests) == [
        "Alice Example met [NAME_1]",
        "[NAME_1] met Bob Sample",
    ]


async def test_the_logs_of_an_embedding_hold_no_value(
    framework: Harness, make: Callable[..., tuple[Any, FakeEmbeddingsApi]]
) -> None:
    gateway, _ = make()
    with capture_logs() as logs:
        async with redaction_scope(known={"NAME": [NAME]}, allow=[f"{NAME} Jr"]):
            await framework.embed_documents(gateway, [f"{NAME} Jr, {NAME}, {EMAIL}, {PHONE}"])
    text = repr(logs)
    for value in (EMAIL, PHONE, NAME):
        assert value not in text
