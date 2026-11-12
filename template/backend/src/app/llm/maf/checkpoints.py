"""MAF workflow checkpoints in Postgres (record 0007, #35 PR 6): MAF's `CheckpointStorage` for one
user, on the `workflow_checkpoint` table (app/llm/core/checkpoints.py).

    storage = gateway.checkpoints(user)
    workflow = WorkflowBuilder(start_executor=..., checkpoint_storage=storage).build()
    async for event in workflow.run(message, stream=True):
        ...
    # later, from the newest checkpoint or any earlier one:
    latest = await storage.get_latest(workflow_name=workflow.name)
    async for event in workflow.run(checkpoint_id=latest.checkpoint_id, stream=True):
        ...

A checkpoint is MAF's own encoding: JSON, with any value that is not JSON-native pickled and
embedded as base64. It is read back with MAF's restricted unpickler, always with an explicit
allow-list (`decode_checkpoint_value(..., allowed_types=frozenset(...))`): MAF's built-in safe set
(primitives, dates, UUIDs and the like), MAF's and openai's own types, and the "module:qualname"
strings in ALLOWED (app/llm/workflow_checkpoint_types.py, yours to edit). Passing None would mean
plain `pickle.loads`, which runs whatever a row names; this module never does. Every checkpoint is
decoded again under the same list before it is saved, so one that could not be resumed fails at save
time, as MAF's own file storage does.

MAF's own documentation of this encoding: the allow-list "is **not** a security boundary";
checkpoint storage must be "access-controlled and not writable by untrusted parties", and data must
come only from the application's own storage. Here it does: rows are written only by this storage,
in the app's database, and scoped to the user they belong to (a checkpoint id from another user is
not found). Never load a checkpoint from anything a client sent.

The encode and decode functions and the "latest" rule are private to agent-framework-core 1.25.0;
the packages are pinned exactly (decision 0002) and a test fails if they move.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from typing import Any, cast

# Private to agent-framework-core 1.25.0 (founder's decision on #35, 11 Nov 2026: use MAF's own
# restricted encoder and decoder, pinned, with a guard test).
from agent_framework._workflows._checkpoint import (
    WorkflowCheckpoint,
    _select_latest_checkpoint,
)
from agent_framework._workflows._checkpoint_encoding import (
    decode_checkpoint_value,
    encode_checkpoint_value,
)
from agent_framework.exceptions import WorkflowCheckpointException
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import workflow_checkpoint_types
from app.llm.core.checkpoints import WorkflowCheckpointRow
from app.llm.core.checkpoints import delete_user as delete_rows
from app.llm.maf.media import refuse_inline_media

__all__ = ["PostgresCheckpointStorage", "allowed_types"]


def allowed_types() -> frozenset[str]:
    """The extra types a checkpoint may hold, from app/llm/workflow_checkpoint_types.py (never
    None: None would mean plain pickle.loads)."""
    return frozenset(workflow_checkpoint_types.ALLOWED)


def _written_by() -> str:
    return f"agent-framework-core {version('agent-framework-core')}"


class PostgresCheckpointStorage:
    """MAF's CheckpointStorage for one user's workflow runs (see the module notes)."""

    def __init__(
        self, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]], user: User
    ) -> None:
        if not isinstance(user.id, str) or not user.id or len(user.id) > 200:
            raise ValueError("the user id must be a non-empty string of at most 200 characters")
        self.sessionmaker = sessionmaker
        self.user_ref = user.id
        self.allowed = allowed_types()

    def _decode(self, document: dict[str, Any]) -> WorkflowCheckpoint:
        decoded = decode_checkpoint_value(document, allowed_types=self.allowed)
        return WorkflowCheckpoint.from_dict(cast("dict[str, Any]", decoded))

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Encode and store a checkpoint; refuses one this storage could not load again, and one
        that holds a file's bytes (InlineMedia: store the file and send its URI, #73)."""
        refuse_inline_media(checkpoint.to_dict())
        try:
            document = cast("dict[str, Any]", encode_checkpoint_value(checkpoint.to_dict()))
            decode_checkpoint_value(document, allowed_types=self.allowed)
        except WorkflowCheckpointException:
            raise
        except Exception as ex:
            raise WorkflowCheckpointException(
                f"Checkpoint {checkpoint.checkpoint_id} cannot be encoded or restored under the "
                "allowed types (app/llm/workflow_checkpoint_types.py); refusing to save."
            ) from ex
        values = {
            "workflow_name": checkpoint.workflow_name,
            "previous_checkpoint_id": checkpoint.previous_checkpoint_id,
            "taken_at": datetime.fromisoformat(checkpoint.timestamp),
            "document": document,
            "written_by": _written_by(),
        }
        async with self.sessionmaker()() as db, db.begin():
            await db.execute(
                insert(WorkflowCheckpointRow)
                .values(user_ref=self.user_ref, checkpoint_id=checkpoint.checkpoint_id, **values)
                .on_conflict_do_update(index_elements=["user_ref", "checkpoint_id"], set_=values)
            )
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """This user's checkpoint `checkpoint_id`; another user's id is not found."""
        async with self.sessionmaker()() as db:
            row = await db.get(WorkflowCheckpointRow, (self.user_ref, checkpoint_id))
        if row is None:
            raise WorkflowCheckpointException(f"No checkpoint found with ID {checkpoint_id}")
        return self._decode(row.document)

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        async with self.sessionmaker()() as db:
            rows = await db.scalars(
                select(WorkflowCheckpointRow)
                .where(
                    WorkflowCheckpointRow.user_ref == self.user_ref,
                    WorkflowCheckpointRow.workflow_name == workflow_name,
                )
                .order_by(WorkflowCheckpointRow.taken_at, WorkflowCheckpointRow.checkpoint_id)
            )
            return [self._decode(row.document) for row in rows]

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]:
        async with self.sessionmaker()() as db:
            ids = await db.scalars(
                select(WorkflowCheckpointRow.checkpoint_id)
                .where(
                    WorkflowCheckpointRow.user_ref == self.user_ref,
                    WorkflowCheckpointRow.workflow_name == workflow_name,
                )
                .order_by(WorkflowCheckpointRow.taken_at, WorkflowCheckpointRow.checkpoint_id)
            )
            return list(ids)

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        """The newest checkpoint, with MAF's own rule for checkpoints that share a timestamp
        (the lineage chain decides, then the id)."""
        mine = (
            WorkflowCheckpointRow.user_ref == self.user_ref,
            WorkflowCheckpointRow.workflow_name == workflow_name,
        )
        async with self.sessionmaker()() as db:
            newest = select(func.max(WorkflowCheckpointRow.taken_at)).where(*mine).scalar_subquery()
            rows = list(
                await db.scalars(
                    select(WorkflowCheckpointRow).where(
                        *mine, WorkflowCheckpointRow.taken_at == newest
                    )
                )
            )
        if not rows:
            return None
        return _select_latest_checkpoint([self._decode(row.document) for row in rows])

    async def delete(self, checkpoint_id: str) -> bool:
        async with self.sessionmaker()() as db, db.begin():
            result = await db.execute(
                delete(WorkflowCheckpointRow).where(
                    WorkflowCheckpointRow.user_ref == self.user_ref,
                    WorkflowCheckpointRow.checkpoint_id == checkpoint_id,
                )
            )
        return bool(getattr(result, "rowcount", 0))

    @staticmethod
    async def delete_user(
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]], user_ref: str
    ) -> int:
        """Delete every stored checkpoint of a user; returns how many there were."""
        async with sessionmaker()() as db, db.begin():
            return await delete_rows(db, user_ref)
