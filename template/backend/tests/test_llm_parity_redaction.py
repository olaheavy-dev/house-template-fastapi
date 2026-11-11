"""What every model framework must do with the text it sends for embeddings (#78, part 3): remove
personal data first, one way, each text on its own, before the budget reserves anything. The
database is real; OpenAI's embeddings endpoint is faked (tests/harness/), so each test runs once per
framework the project chose and imports none.

The values are synthetic: the email and phone are from reserved or never-allocated ranges, the name
is made up. A placeholder is [KIND_n]; "[EMAIL_1]" is how every email below is sent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import anyio
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from app.llm.core.budget import LlmCall
from app.llm.core.redaction import RedactionFailed, Redactor, redaction_scope
from app.settings import Settings
from tests.evals.redaction_corpus import MUST_NOT_CHANGE
from tests.harness import FakeEmbeddingsApi, Harness

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 3)
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
