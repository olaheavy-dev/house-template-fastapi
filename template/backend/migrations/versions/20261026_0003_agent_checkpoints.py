"""Agent checkpoints: LangGraph's Postgres checkpointer tables

Created by the library's own migration list (app/llm/checkpoints.py), outside a transaction
because it builds indexes CONCURRENTLY. The tables are new, so no existing rows are affected.
Downgrading drops every saved agent thread.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from app.llm import checkpoints

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        if op.get_context().as_sql:  # `alembic upgrade heads --sql`: no database to ask
            for statement in checkpoints.migration_sql():
                op.execute(statement)
        else:
            checkpoints.apply_migrations(op.get_bind())


def downgrade() -> None:
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
        op.drop_table(table)
