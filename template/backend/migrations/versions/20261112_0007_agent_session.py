"""Stored agent sessions: one JSON document per user, conversation and agent (#35, decision 0004)

Created with any model gateway, not only MAF's (which uses it, app/llm/maf/sessions.py): a project
that switches from LangChain to MAF by `copier update` then has the table, which an empty revision
would leave it without. A LangChain project keeps the table empty. The table is new, so no
existing rows are affected. Downgrading drops every stored session.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-12
"""

from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401 - autogenerate writes VECTOR columns as pgvector.sqlalchemy.*
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_session",
        sa.Column("user_ref", sa.String(length=200), nullable=False),
        sa.Column("conversation_id", sa.String(length=200), nullable=False),
        sa.Column("agent", sa.String(length=200), nullable=False),
        sa.Column("document", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("written_by", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "user_ref", "conversation_id", "agent", name=op.f("pk_agent_session")
        ),
    )


def downgrade() -> None:
    op.drop_table("agent_session")
