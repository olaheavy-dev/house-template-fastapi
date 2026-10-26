"""Per-user model budget: each user's reserved plus settled spend per UTC day

A new table, so no existing rows are affected; llm_call.user_reserved is false for calls made
before it (none of them reserved against a user's cap). Rows appear only for calls tagged with a
user while LLM_USER_DAILY_BUDGET_USD is set. Downgrading drops the per-user history; the day
totals stay.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - autogenerate writes VECTOR columns as pgvector.sqlalchemy.*
import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_spend_user_day",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("user_ref", sa.String(length=200), nullable=False),
        sa.Column("committed_micro_usd", sa.BigInteger(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("day", "user_ref", name=op.f("pk_llm_spend_user_day")),
    )
    op.add_column(
        "llm_call",
        sa.Column("user_reserved", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_call", "user_reserved")
    op.drop_table("llm_spend_user_day")
