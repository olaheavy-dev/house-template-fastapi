"""Stored agent conversations (decision 0004): one JSON document per user, conversation and agent.

The row is framework-neutral data; an adapter decides what the document is (app/llm/maf/sessions.py
stores MAF's `AgentSession.to_dict()`). The key is the user, the conversation and the agent: an
adapter derives it from the current user on the server, so a caller can never name another user's
thread. `version` is checked on every save, so two turns that ran at the same time cannot silently
overwrite each other: the second save is refused (409) and the caller reloads.

The document holds conversation text, so a user's rows are deleted with the user's data
(`delete_user`). Text is stored as the user wrote it (redaction applies to what is sent to the
provider, not to what we keep); images are stored as a pointer to a file, never their bytes (#73).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, delete, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.problems import Problem

__all__ = ["AgentSessionRow", "SessionConflict", "delete_user"]


class AgentSessionRow(Base):
    """One agent's conversation for one user, as the adapter's JSON document."""

    __tablename__ = "agent_session"

    user_ref: Mapped[str] = mapped_column(String(200), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    agent: Mapped[str] = mapped_column(String(200), primary_key=True)
    document: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Bumped on every save; a save that read an older version is refused (SessionConflict).
    version: Mapped[int] = mapped_column(Integer)
    # The framework package and version that wrote the document: a newer framework's sessions may
    # not be readable by an older one, so a rollback can find what it cannot load.
    written_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SessionConflict(Problem):
    """The conversation was saved by another turn since this one loaded it (409): reload and
    retry. Nothing of the conversation is in the response."""

    def __init__(self) -> None:
        super().__init__(
            409, detail="This conversation changed while it was being answered. Try again."
        )


async def delete_user(session: AsyncSession, user_ref: str) -> int:
    """Delete every stored conversation of a user (with the rest of their data); returns how many
    rows went. The caller commits."""
    result = await session.execute(
        delete(AgentSessionRow).where(AgentSessionRow.user_ref == user_ref)
    )
    return int(getattr(result, "rowcount", 0) or 0)
