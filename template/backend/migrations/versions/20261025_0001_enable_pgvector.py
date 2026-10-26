"""Enable pgvector

Extensions are outside what autogenerate sees, so this one is written by hand.

The first of the template's own migrations: they form the "template" branch, separate from the
project's "app" branch (see the app root revision), so a template update that ships a migration
never collides with the project's.

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = ("template",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Left in place: the extension may predate this migration (a managed database often has it
    # installed by an admin), and dropping it would take other schemas' vector columns with it.
    pass
