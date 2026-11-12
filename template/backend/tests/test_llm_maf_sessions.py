"""MAF sessions in Postgres (#35 PR 5, decision 0004): what is stored, for whom, and what is
refused. The shared cases (an image in a thread across a restart, another user's conversation)
are in test_llm_parity_threads.py. The values are synthetic."""

from __future__ import annotations

import json
from datetime import date

import pytest
from agent_framework import AgentSession, Content, Message
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm.core.sessions import AgentSessionRow, SessionConflict
from app.llm.maf.gateway import Gateway
from app.llm.maf.sessions import InlineMedia
from app.settings import Settings
from tests.harness import Reply
from tests.harness.maf import HARNESS, ScriptedClient

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 12)
EMAIL = "priya.shah@example.com"
ALICE, BOB = User(id="u_alice"), User(id="u_bob")


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def gateway(settings: Settings, sessions: async_sessionmaker[AsyncSession]) -> Gateway:
    built = Gateway(
        settings.model_copy(update={"llm_model": "anthropic:claude-sonnet-5-5"}), lambda: sessions
    )
    built.meter.cap, built.meter.today = 1_000_000, lambda: DAY
    return built


def scripted(gateway: Gateway, *replies: str) -> ScriptedClient:
    HARNESS.install_scripted(gateway, [Reply(text) for text in replies])
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    return client


async def turn(gateway: Gateway, who: User, conversation: str, text: str) -> str:
    agent = gateway.agent([], name="helper")
    session = await gateway.sessions.load(who, conversation, agent)
    response = await agent.run(text, session=session)
    await gateway.sessions.save(who, conversation, agent, session)
    return response.text


async def stored(db: AsyncSession, who: User, conversation: str) -> AgentSessionRow:
    row = await db.get(AgentSessionRow, (who.id, conversation, "helper"))
    assert row is not None
    return row


async def test_the_stored_history_keeps_the_originals_while_each_request_has_placeholders(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    client = scripted(gateway, "I will write to [EMAIL_1].", "Done.")
    assert await turn(gateway, ALICE, "c1", f"Write to {EMAIL}") == f"I will write to {EMAIL}."
    await turn(gateway, ALICE, "c1", "Thanks")
    # Both requests carried the placeholder only, the second one with the stored history.
    for request in client.seen:
        text = json.dumps([message.to_dict() for message in request], default=str)
        assert EMAIL not in text
    assert "[EMAIL_1]" in json.dumps([m.to_dict() for m in client.seen[1]], default=str)
    # What we keep is what the user wrote and the answer they saw.
    row = await stored(db_session, ALICE, "c1")
    document = json.dumps(row.document)
    assert document.count(EMAIL) == 2
    assert "[EMAIL_1]" not in document
    assert row.version == 2
    assert row.written_by.startswith("agent-framework-core ")


async def test_a_second_turn_saved_over_a_newer_one_is_a_409_and_changes_nothing(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    scripted(gateway, "one", "two", "three")
    await turn(gateway, ALICE, "c2", "first")
    agent = gateway.agent([], name="helper")
    early = await gateway.sessions.load(ALICE, "c2", agent)
    late = await gateway.sessions.load(ALICE, "c2", agent)
    await agent.run("second", session=early)
    await agent.run("third", session=late)
    await gateway.sessions.save(ALICE, "c2", agent, early)
    with pytest.raises(SessionConflict) as raised:
        await gateway.sessions.save(ALICE, "c2", agent, late)
    assert raised.value.status == 409
    document = json.dumps((await stored(db_session, ALICE, "c2")).document)
    assert "second" in document
    assert "third" not in document


async def test_two_new_conversations_saved_at_once_keep_the_first(gateway: Gateway) -> None:
    scripted(gateway, "a", "b")
    agent = gateway.agent([], name="helper")
    one = await gateway.sessions.load(ALICE, "c3", agent)
    two = await gateway.sessions.load(ALICE, "c3", agent)
    await agent.run("x", session=one)
    await agent.run("y", session=two)
    await gateway.sessions.save(ALICE, "c3", agent, one)
    with pytest.raises(SessionConflict):
        await gateway.sessions.save(ALICE, "c3", agent, two)


async def test_deleting_a_users_data_deletes_their_sessions_only(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    scripted(gateway, "a", "b", "c")
    await turn(gateway, ALICE, "c4", "hi")
    await turn(gateway, ALICE, "c5", "hi")
    await turn(gateway, BOB, "c4", "hi")
    assert await gateway.sessions.delete_user(ALICE.id) == 2
    left = list(await db_session.scalars(select(AgentSessionRow.user_ref)))
    assert left == [BOB.id]


async def test_a_file_sent_as_bytes_is_refused_and_nothing_is_stored(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    # #73: a thread keeps a pointer to a stored file, never its bytes.
    scripted(gateway, "a room")
    agent = gateway.agent([], name="helper")
    session = await gateway.sessions.load(ALICE, "c6", agent)
    image = Content.from_data(b"\x89PNG\r\n", media_type="image/png")
    await agent.run([Message("user", [image, "Which room?"])], session=session)
    with pytest.raises(InlineMedia):
        await gateway.sessions.save(ALICE, "c6", agent, session)
    assert await db_session.get(AgentSessionRow, (ALICE.id, "c6", "helper")) is None


async def test_a_stored_document_builds_only_the_types_maf_registered(
    gateway: Gateway, db_session: AsyncSession
) -> None:
    # A row someone edited to name a class: MAF rebuilds only registered types (Message) and
    # leaves any other `type` as plain data. Nothing is imported or called.
    scripted(gateway, "a")
    await turn(gateway, ALICE, "c7", "hi")
    planted = {"type": "os.system", "args": ["echo hacked"]}
    row = await stored(db_session, ALICE, "c7")
    document = dict(row.document)
    document["state"] = {**document["state"], "planted": planted}
    await db_session.execute(
        update(AgentSessionRow)
        .where(AgentSessionRow.user_ref == ALICE.id, AgentSessionRow.conversation_id == "c7")
        .values(document=document)
    )
    loaded = await gateway.sessions.load(ALICE, "c7", gateway.agent([], name="helper"))
    assert loaded.state["planted"] == planted
    messages = loaded.state["in_memory"]["messages"]
    assert all(isinstance(message, Message) for message in messages)


async def test_an_agent_without_a_stable_name_cannot_keep_a_conversation(
    gateway: Gateway,
) -> None:
    with pytest.raises(ValueError, match="stable name"):
        await gateway.sessions.load(ALICE, "c8", gateway.agent([]))


async def test_only_a_session_this_store_loaded_can_be_saved(gateway: Gateway) -> None:
    agent = gateway.agent([], name="helper")
    with pytest.raises(ValueError, match="loaded"):
        await gateway.sessions.save(ALICE, "c9", agent, AgentSession())


def test_maf_restores_only_registered_state_types_in_the_pinned_core() -> None:
    # The store relies on this behaviour of agent-framework-core 1.25.0 (from_dict leaves an
    # unregistered `type` as a dict): a version that changes it must fail here.
    restored = AgentSession.from_dict(
        {"session_id": "s", "state": {"x": {"type": "builtins.eval", "value": "1"}}}
    )
    assert restored.state["x"] == {"type": "builtins.eval", "value": "1"}
    with_message = AgentSession.from_dict(
        {"session_id": "s", "state": {"m": Message("user", ["hi"]).to_dict()}}
    )
    assert isinstance(with_message.state["m"], Message)
