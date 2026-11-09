"""Turning the gateway on in a database that ran without it (#30).

With `llm: none` the template's migrations 0002 to 0005 stay in the chain but do nothing, so such a
database is stamped at 0005 with no gateway tables. The recipe for adding them later is in the
README ("Upgrading to 0.8.0"): `alembic stamp 0001`, then `alembic upgrade heads`.
"""

from __future__ import annotations

from alembic import command
from sqlalchemy import create_engine, inspect

from tests import db

GATEWAY_TABLES = {
    "llm_call",
    "llm_spend_day",
    "llm_spend_user_day",
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}


def tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_llm_tables_can_be_added_to_a_database_that_ran_without_them(database_url: str) -> None:
    config = db.alembic_config(database_url)
    try:
        # A database as `llm: none` leaves it: template revisions up to 0005 applied (each did
        # nothing), no gateway tables. Made here by removing the real tables and re-stamping.
        command.downgrade(config, "0001")
        assert not tables(database_url) & GATEWAY_TABLES
        command.stamp(config, "0005")
        assert not tables(database_url) & GATEWAY_TABLES

        # The recipe.
        command.stamp(config, "0001")
        command.upgrade(config, "heads")

        assert tables(database_url) >= GATEWAY_TABLES
        command.check(config)
    finally:
        # Whatever happened, leave the session database at heads for the tests after this one.
        command.upgrade(config, "heads")


def test_downgrade_to_0001_leaves_the_project_branch_alone(database_url: str) -> None:
    # The README says to drop the gateway's tables with `alembic downgrade 0001`: that removes the
    # template's revisions above 0001 and does not touch the project's branch.
    config = db.alembic_config(database_url)
    try:
        command.downgrade(config, "0001")
        engine = create_engine(database_url)
        try:
            with engine.connect() as connection:
                rows = connection.exec_driver_sql("SELECT version_num FROM alembic_version")
                versions = {row[0] for row in rows}
        finally:
            engine.dispose()
        assert versions == {"0001", "app_root"}
        assert not tables(database_url) & GATEWAY_TABLES
    finally:
        command.upgrade(config, "heads")
