"""Model budget: daily spend and one row per model call

The tables are new, so no existing rows are affected. Downgrading drops the spend history.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - autogenerate writes VECTOR columns as pgvector.sqlalchemy.*
import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_call",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("prompt", sa.String(length=200), nullable=True),
        sa.Column("prompt_sha", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reserved_micro_usd", sa.BigInteger(), nullable=False),
        sa.Column("cost_micro_usd", sa.BigInteger(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "status IN ('reserved', 'settled', 'failed')", name=op.f("ck_llm_call_status")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_call")),
    )
    op.create_index(op.f("ix_llm_call_day"), "llm_call", ["day"], unique=False)
    op.create_table(
        "llm_spend_day",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("committed_micro_usd", sa.BigInteger(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("day", name=op.f("pk_llm_spend_day")),
    )


def downgrade() -> None:
    op.drop_table("llm_spend_day")
    op.drop_index(op.f("ix_llm_call_day"), table_name="llm_call")
    op.drop_table("llm_call")
