"""Stored workflow checkpoints (record 0007): one row per user and checkpoint.

The row is framework-neutral; an adapter decides what the document is (app/llm/maf/checkpoints.py
stores MAF's encoded `WorkflowCheckpoint`). Rows are scoped to a user, derived on the server: a
user can only list, load or delete their own checkpoints, so a checkpoint id from someone else is
simply not found. The document holds conversation text and workflow state, so a user's rows are
deleted with the user's data (`delete_user`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, delete, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

__all__ = ["WorkflowCheckpointRow", "delete_user"]


class WorkflowCheckpointRow(Base):
    """One saved checkpoint of a workflow run, for one user, as the adapter's encoded document."""

    __tablename__ = "workflow_checkpoint"
    __table_args__ = (
        Index(
            "ix_workflow_checkpoint_user_ref_workflow_name_taken_at",
            "user_ref",
            "workflow_name",
            "taken_at",
        ),
    )

    user_ref: Mapped[str] = mapped_column(String(200), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    workflow_name: Mapped[str] = mapped_column(String(200))
    previous_checkpoint_id: Mapped[str | None] = mapped_column(String(100))
    # When the checkpoint was taken (the framework's own timestamp), for "latest" and listing.
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    document: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # The framework package and version that wrote the document (a rollback can find what it
    # cannot load).
    written_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


async def delete_user(session: AsyncSession, user_ref: str) -> int:
    """Delete every stored checkpoint of a user; returns how many rows went. The caller
    commits."""
    result = await session.execute(
        delete(WorkflowCheckpointRow).where(WorkflowCheckpointRow.user_ref == user_ref)
    )
    return int(getattr(result, "rowcount", 0) or 0)
