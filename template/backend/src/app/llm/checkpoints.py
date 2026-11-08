"""Moved (#33): LangGraph's Postgres checkpointer is in app/llm/langchain/checkpoints.py
(MIGRATIONS, TABLES, apply_migrations, migration_sql, open_checkpointer). Until 0.8.0 the old names
still answer here, with a DeprecationWarning, and are the same objects as the new ones.

A project's own Alembic revision that calls apply_migrations must import the new module: the old
path warns, and under pytest's `filterwarnings = error` the warning stops every database test.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.langchain import checkpoints as _checkpoints

if TYPE_CHECKING:
    from app.llm.langchain.checkpoints import MIGRATIONS as MIGRATIONS
    from app.llm.langchain.checkpoints import TABLES as TABLES
    from app.llm.langchain.checkpoints import apply_migrations as apply_migrations
    from app.llm.langchain.checkpoints import migration_sql as migration_sql
    from app.llm.langchain.checkpoints import open_checkpointer as open_checkpointer

__all__ = ["MIGRATIONS", "TABLES", "apply_migrations", "migration_sql", "open_checkpointer"]

_MOVED = frozenset(__all__)

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            warnings.warn(
                f"app.llm.checkpoints.{name} moved to {_checkpoints.__name__}.{name}: import it "
                "from there (the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(_checkpoints, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
