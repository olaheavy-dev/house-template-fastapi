"""The project's own migrations start here

An empty root for the "app" branch. The template's migrations live on the "template" branch
(0001 onwards) and arrive through template updates; the project's own live on this one, so the
two never share a parent and an update can't leave two competing heads on one branch. Upgrade
both with `alembic upgrade heads` (make migrate); `make migration` writes new revisions here.

A project migration that needs a template table (a foreign key to llm_call, say) declares
depends_on with that template revision, so Alembic applies it first.

Revision ID: app_root
Revises:
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "app_root"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = ("app",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
