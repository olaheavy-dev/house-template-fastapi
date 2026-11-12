"""Turning the gateway on in a database that ran without it (#30).

With `llm: none` the template's migrations 0002 to 0008 stay in the chain but do nothing, so such a
database is stamped at 0008 with no gateway tables. The recipe for adding them later is in the
README ("Upgrading to 0.8.0"): `alembic stamp 0001`, then `alembic upgrade heads`. A database that
had the tables, ran a later release with `llm: none` (0006 to 0008 did nothing there) and turns the
gateway back on lacks 0006's column and the 0007 and 0008 tables: the recipe for that is `alembic
stamp 0005`, then `alembic upgrade heads` (see "Upgrading", decision 0005).
"""

from __future__ import annotations

from alembic import command
from sqlalchemy import create_engine, inspect

from tests import db

GATEWAY_TABLES = {
    "agent_session",
    "workflow_checkpoint",
    "llm_call",
    "llm_spend_day",
    "llm_spend_user_day",
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}


def llm_call_columns(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return {column["name"] for column in inspect(engine).get_columns("llm_call")}
    finally:
        engine.dispose()


def tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_llm_tables_can_be_added_to_a_database_that_ran_without_them(database_url: str) -> None:
    config = db.alembic_config(database_url)
    try:
        # A database as `llm: none` leaves it: template revisions up to 0008 applied (each did
        # nothing), no gateway tables. Made here by removing the real tables and re-stamping.
        command.downgrade(config, "0001")
        assert not tables(database_url) & GATEWAY_TABLES
        command.stamp(config, "0008")
        assert not tables(database_url) & GATEWAY_TABLES

        # The recipe.
        command.stamp(config, "0001")
        command.upgrade(config, "heads")

        assert tables(database_url) >= GATEWAY_TABLES
        assert "redacted_count" in llm_call_columns(database_url)
        command.check(config)
    finally:
        # Whatever happened, leave the session database at heads for the tests after this one.
        command.upgrade(config, "heads")


def versions_in(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            rows = connection.exec_driver_sql("SELECT version_num FROM alembic_version")
            return {row[0] for row in rows}
    finally:
        engine.dispose()


def test_downgrade_to_0001_removes_the_gateway_tables(database_url: str) -> None:
    # The README says to drop the gateway's tables with `alembic downgrade 0001`. What is
    # guaranteed: 0001 stays stamped, no gateway revision stays stamped, and the tables are gone.
    # Not guaranteed, so not asserted: that your own migrations stay stamped. One that declares
    # `depends_on` a gateway revision counts as its child in Alembic's graph and is downgraded too.
    config = db.alembic_config(database_url)
    try:
        command.downgrade(config, "0001")
        after = versions_in(database_url)
        assert "0001" in after
        assert not after & {"0002", "0003", "0004", "0005", "0006", "0007", "0008"}
        assert not tables(database_url) & GATEWAY_TABLES
    finally:
        command.upgrade(config, "heads")


def test_0006_adds_and_drops_the_redaction_count_column(database_url: str) -> None:
    config = db.alembic_config(database_url)
    try:
        assert "redacted_count" in llm_call_columns(database_url)
        command.downgrade(config, "0005")
        assert "redacted_count" not in llm_call_columns(database_url)
        # Only that column went: the table and the earlier revisions' columns stay.
        assert {"id", "user_reserved", "feature"} <= llm_call_columns(database_url)
        assert "0005" in versions_in(database_url)
        command.upgrade(config, "heads")
        assert "redacted_count" in llm_call_columns(database_url)
        command.check(config)
    finally:
        command.upgrade(config, "heads")


def test_a_database_stamped_past_0006_while_the_gateway_was_off_gets_the_column_with_stamp_0005(
    database_url: str,
) -> None:
    # The gateway ran up to 0005 (tables, no redacted_count). The project then updated the template
    # with `llm: none`, where 0006 to 0008 do nothing, so its database is stamped at 0008 and the
    # column and the session and checkpoint tables are missing. Turning the gateway back on and
    # running `make migrate` would do nothing.
    config = db.alembic_config(database_url)
    try:
        command.downgrade(config, "0005")
        assert tables(database_url) >= GATEWAY_TABLES - {"agent_session", "workflow_checkpoint"}
        command.stamp(config, "0008")
        command.upgrade(config, "heads")
        assert "redacted_count" not in llm_call_columns(database_url)  # the gap
        assert not tables(database_url) & {"agent_session", "workflow_checkpoint"}

        # The recipe: stamp 0005, then migrate. 0006 to 0008 now run for real.
        command.stamp(config, "0005")
        command.upgrade(config, "heads")

        assert "redacted_count" in llm_call_columns(database_url)
        assert tables(database_url) >= GATEWAY_TABLES
        command.check(config)
    finally:
        command.upgrade(config, "heads")
