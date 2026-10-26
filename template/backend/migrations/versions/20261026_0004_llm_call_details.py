"""Model call details: reasoning and cache tokens, latency, feature and user tags; the
daily budget warning flag

Every new column is nullable: calls recorded before this migration keep NULL for them (their
details were never captured). The feature index is built with a plain CREATE INDEX, which blocks
writes to llm_call while it builds; for a large table, build it CONCURRENTLY by hand first.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - autogenerate writes VECTOR columns as pgvector.sqlalchemy.*
import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("llm_call", sa.Column("reasoning_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_call", sa.Column("cache_read_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_call", sa.Column("cache_write_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_call", sa.Column("latency_ms", sa.Integer(), nullable=True))
    op.add_column("llm_call", sa.Column("feature", sa.String(length=100), nullable=True))
    op.add_column("llm_call", sa.Column("user_ref", sa.String(length=200), nullable=True))
    op.create_index(op.f("ix_llm_call_feature"), "llm_call", ["feature"], unique=False)
    op.add_column(
        "llm_spend_day",
        sa.Column("warned", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_spend_day", "warned")
    op.drop_index(op.f("ix_llm_call_feature"), table_name="llm_call")
    op.drop_column("llm_call", "user_ref")
    op.drop_column("llm_call", "feature")
    op.drop_column("llm_call", "latency_ms")
    op.drop_column("llm_call", "cache_write_tokens")
    op.drop_column("llm_call", "cache_read_tokens")
    op.drop_column("llm_call", "reasoning_tokens")
