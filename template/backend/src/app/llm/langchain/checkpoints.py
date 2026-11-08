"""LangGraph's Postgres checkpointer: agent threads that survive restarts.

Its tables are created by Alembic, not at startup: the library's setup() builds indexes
CONCURRENTLY (impossible inside a transaction) and would race when several workers start at once.
apply_migrations runs the library's own migration list the way setup() does, recording each
version in checkpoint_migrations, so the two never disagree. When an upgrade of
langgraph-checkpoint-postgres adds a migration, a test fails until a new Alembic revision calls
apply_migrations again.

The checkpointer gets its own small psycopg pool (the library needs autocommit, no prepared
statements and dict rows), opened in the lifespan; it holds no connection until an agent runs.

Saved threads load strictly (#36): see serializer().
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import make_url, text
from sqlalchemy.engine import Connection

from app.llm import checkpoint_types
from app.settings import Settings

MIGRATIONS: list[str] = BasePostgresSaver.MIGRATIONS
# The library's tables: Alembic's autogenerate leaves them alone (migrations/env.py).
TABLES = frozenset(
    {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
)


def apply_migrations(connection: Connection) -> None:
    """Bring the checkpoint tables up to the installed library's version. Call it inside
    op.get_context().autocommit_block(): some of the library's migrations can't run in a
    transaction."""
    connection.execute(text(MIGRATIONS[0]))
    done = connection.scalar(text("SELECT max(v) FROM checkpoint_migrations"))
    for version in range(0 if done is None else done + 1, len(MIGRATIONS)):
        connection.execute(text(MIGRATIONS[version]))
        connection.execute(
            text("INSERT INTO checkpoint_migrations (v) VALUES (:v)"), {"v": version}
        )


def migration_sql() -> list[str]:
    """Every migration and its version row, for a fresh database: what `alembic upgrade heads
    --sql` prints, since offline there is no database to ask which have already run."""
    statements = []
    for version, sql in enumerate(MIGRATIONS):
        insert = f"INSERT INTO checkpoint_migrations (v) VALUES ({version})"  # noqa: S608 - an int
        statements += [sql, insert]
    return statements


def serializer(allowed: Iterable[type] = ()) -> JsonPlusSerializer:
    """LangGraph's serializer, loading strictly: a saved object is rebuilt only if its class is on
    LangGraph's safe list (messages, dates, UUIDs, `Send` and so on) or in `allowed`, which adds to
    that list. Any other object comes back as its plain data and LangGraph logs "Blocked
    deserialization of <module>.<name>" once per class and process. LangGraph's default loader
    imports and calls whatever a row names (GHSA-g48c-2wqr-h844).

    Set here, not with LANGGRAPH_STRICT_MSGPACK: LangGraph reads that variable once, at import, and
    backend/.env never reaches the process environment under `make dev` or the tests, so dev and
    tests would load differently from production."""
    # An empty tuple means "the safe list only", the same as None.
    return JsonPlusSerializer(allowed_msgpack_modules=tuple(allowed) or None)


async def open_checkpointer(settings: Settings) -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    # libpq takes the URL without SQLAlchemy's "+psycopg" driver suffix.
    url = make_url(settings.database_url.get_secret_value()).set(drivername="postgresql")
    kwargs: dict[str, Any] = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}
    pool: AsyncConnectionPool = AsyncConnectionPool(
        url.render_as_string(hide_password=False),
        min_size=0,
        max_size=settings.agent_pool_size,
        kwargs=kwargs,
        open=False,
    )
    await pool.open()
    # checkpoint_types.ALLOWED is read here, at call time, so a project or a test can change it.
    saver = AsyncPostgresSaver(pool, serde=serializer(checkpoint_types.ALLOWED))  # type: ignore[arg-type]
    return saver, pool
