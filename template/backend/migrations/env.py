"""Alembic environment: migrations run on the app's own settings and metadata.

The URL comes from DATABASE_URL (app.settings), unless the caller put one in
Config.attributes, as the tests do. `alembic check`
(make check) fails when a model has no migration.
"""

from __future__ import annotations

import asyncio
import logging

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.models
from app.llm import checkpoints
from app.settings import get_settings

config = context.config

# No alembic.ini, so no logging config from it: show Alembic's progress lines on the command line.
logging.getLogger("alembic").setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(format="%(message)s")
target_metadata = app.models.Base.metadata


def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    # LangGraph's checkpoint tables are created by its own migrations (app/llm/checkpoints.py),
    # so autogenerate must neither drop them nor report them as unmodelled.
    return not (type_ == "table" and name in checkpoints.TABLES)


def database_url() -> str:
    # Config.attributes, not sqlalchemy.url: there is no alembic.ini to hold main options.
    url: str | None = config.attributes.get("database_url")
    return url or get_settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Write the SQL to stdout (`alembic upgrade heads --sql`) instead of running it."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, include_name=include_name
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # A migration that waits on a lock queues every later query on that table behind it: give up
    # after 5 s instead (retry the rollout when traffic is quieter). Statements themselves may run
    # long (building an index on a big table), so their limit is generous.
    engine = create_async_engine(
        database_url(),
        poolclass=pool.NullPool,
        connect_args={"options": "-c lock_timeout=5s -c statement_timeout=15min"},
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
