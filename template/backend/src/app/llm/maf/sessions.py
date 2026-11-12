"""MAF agent sessions in Postgres (decision 0004, #35 PR 5): one row per user, conversation and
agent (app/llm/core/sessions.py), holding the whole `AgentSession` as JSON.

    agent = gateway.agent([search_courses], instructions=prompt.text, name="courses")
    session = await gateway.sessions.load(user, conversation_id, agent)
    with gateway.call_options(user, "answer", prompt):
        response = await agent.run(question, session=session)
    await gateway.sessions.save(user, conversation_id, agent, session)

What is stored is MAF's own `AgentSession.to_dict()`: its history (MAF's InMemoryHistoryProvider,
which the agent adds for a session when no history provider is configured, keeps the messages in
`session.state`), and anything else in its state, such as pending tool approvals. It is read back
with `AgentSession.from_dict`, which rebuilds only the types registered with MAF
(`register_state_type`; `Message` is) and leaves anything else as plain data: no pickle, no import
by name (agent-framework-core 1.25.0; the founder's MAF lab stores sessions the same way).

The key is derived here from the current user, the conversation id and the agent's name; it is never
taken from a caller (a caller-supplied thread id would let one user read another's thread). Confirm
the user owns the conversation id before you use it. Give each agent a stable `name`: a session
belongs to one agent, and a renamed agent starts its conversations again.

The history keeps the user's own text and the restored answers: redaction applies to what each model
call sends, not to what the session stores. An image must be a pointer to a stored file
(`Content.from_uri`), never its bytes (#73): `save` refuses inline data. Two turns of one
conversation that run at the same time cannot overwrite each other: the second `save` raises
SessionConflict (409); reload and run again. Delete a user's sessions with the rest of their data
(`delete_user`). History is not trimmed yet (#46): a long conversation grows its row.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

from agent_framework import AgentSession
from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm.core.sessions import AgentSessionRow, SessionConflict
from app.llm.core.sessions import delete_user as delete_rows

__all__ = ["InlineMedia", "PostgresSessionStore"]

_MAX_KEY = 200  # the columns' length


class InlineMedia(ValueError):
    """A session holds a file's bytes (`Content.from_data`): store the file and send its URI."""


def _written_by() -> str:
    return f"agent-framework-core {version('agent-framework-core')}"


def _agent_name(agent: Any) -> str:
    name = getattr(agent, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("give the agent a stable name: gateway.agent(..., name='...')")
    return name


def _check_key(part: str, what: str) -> str:
    if not isinstance(part, str) or not part or len(part) > _MAX_KEY:
        raise ValueError(f"the {what} must be a non-empty string of at most {_MAX_KEY} characters")
    return part


def _refuse_inline_bytes(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "data":
            raise InlineMedia(
                "a session must not store a file's bytes: store the file and send its URI (#73)"
            )
        for item in value.values():
            _refuse_inline_bytes(item)
    elif isinstance(value, list):
        for item in value:
            _refuse_inline_bytes(item)


class PostgresSessionStore:
    """Loads and saves MAF agent sessions for the current user (see the module notes)."""

    def __init__(self, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]) -> None:
        self.sessionmaker = sessionmaker
        # The version each loaded session was read at, for the conflict check on save. Kept beside
        # the session (not in its state, which MAF serializes) and forgotten with it.
        self._versions: weakref.WeakKeyDictionary[AgentSession, int] = weakref.WeakKeyDictionary()

    @staticmethod
    def _key(user: User, conversation_id: str, agent: Any) -> tuple[str, str, str]:
        return (
            _check_key(user.id, "user id"),
            _check_key(conversation_id, "conversation id"),
            _check_key(_agent_name(agent), "agent name"),
        )

    async def load(self, user: User, conversation_id: str, agent: Any) -> AgentSession:
        """The user's session with this agent in this conversation, or a new one."""
        key = self._key(user, conversation_id, agent)
        async with self.sessionmaker()() as db:
            row = await db.get(AgentSessionRow, key)
        if row is None:
            session: AgentSession = agent.create_session()
            self._versions[session] = 0
            return session
        session = AgentSession.from_dict(row.document)
        self._versions[session] = row.version
        return session

    async def save(
        self, user: User, conversation_id: str, agent: Any, session: AgentSession
    ) -> None:
        """Store the session after a run. Raises SessionConflict when another turn saved this
        conversation since `session` was loaded, and InlineMedia when it holds a file's bytes."""
        key = self._key(user, conversation_id, agent)
        expected = self._versions.get(session)
        if expected is None:
            raise ValueError("save a session this store loaded (gateway.sessions.load)")
        document = session.to_dict()
        _refuse_inline_bytes(document)
        user_ref, conversation, name = key
        async with self.sessionmaker()() as db, db.begin():
            if expected == 0:
                inserted = await db.execute(
                    insert(AgentSessionRow)
                    .values(
                        user_ref=user_ref,
                        conversation_id=conversation,
                        agent=name,
                        document=document,
                        version=1,
                        written_by=_written_by(),
                    )
                    .on_conflict_do_nothing()
                    .returning(AgentSessionRow.version)
                )
                saved = inserted.scalar_one_or_none() is not None
            else:
                updated = await db.execute(
                    update(AgentSessionRow)
                    .where(
                        AgentSessionRow.user_ref == user_ref,
                        AgentSessionRow.conversation_id == conversation,
                        AgentSessionRow.agent == name,
                        AgentSessionRow.version == expected,
                    )
                    .values(
                        document=document,
                        version=expected + 1,
                        written_by=_written_by(),
                        updated_at=func.now(),
                    )
                    .returning(AgentSessionRow.version)
                )
                saved = updated.scalar_one_or_none() is not None
        if not saved:
            raise SessionConflict
        self._versions[session] = expected + 1

    async def delete_user(self, user_ref: str) -> int:
        """Delete every stored session of a user; returns how many there were."""
        async with self.sessionmaker()() as db, db.begin():
            return await delete_rows(db, user_ref)
