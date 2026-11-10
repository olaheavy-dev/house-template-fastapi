"""Redaction count: how many values personal-data redaction replaced in each model call (#78)

One new nullable column. Calls recorded before it keep NULL, which also means "redaction was
off"; 0 means redaction ran and found nothing. It is a count, never a value. Downgrading drops
the counts.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-10
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - autogenerate writes VECTOR columns as pgvector.sqlalchemy.*
import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("llm_call", sa.Column("redacted_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_call", "redacted_count")
