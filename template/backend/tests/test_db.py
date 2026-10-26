from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.script import ScriptDirectory
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionDep, create_engine, ping
from app.main import create_app
from app.settings import Settings
from tests import db

pytestmark = pytest.mark.anyio


async def test_migrations_enable_pgvector(db_session: AsyncSession) -> None:
    version = await db_session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    assert version is not None
    distance = await db_session.scalar(text("SELECT '[1,0]'::vector <=> '[0,1]'::vector"))
    assert distance == pytest.approx(1.0)


def test_every_model_has_a_migration(database_url: str) -> None:
    # The same check `alembic check` makes: autogenerate finds nothing new.
    command.check(db.alembic_config(database_url))


def test_migrations_downgrade_and_upgrade_cleanly(database_url: str) -> None:
    config = db.alembic_config(database_url)
    command.downgrade(config, "base")
    command.upgrade(config, "heads")


async def test_requests_get_a_session(app: FastAPI, db_client: httpx.AsyncClient) -> None:
    @app.get("/one")
    async def one(session: SessionDep) -> int:
        return int(await session.scalar(text("SELECT 1")) or 0)

    assert (await db_client.get("/one")).json() == 1


async def test_db_client_points_the_app_itself_at_the_test_database(
    app: FastAPI, db_client: httpx.AsyncClient
) -> None:
    # Code that uses app.state (readiness, background work) must reach the test database too,
    # never the developer's local one.
    assert (await db_client.get("/health/ready")).status_code == 200

    @app.post("/side-write")
    async def side_write() -> None:
        async with app.state.sessionmaker() as session:
            await session.execute(text("CREATE TABLE side_effect (id int)"))
            await session.commit()

    await db_client.post("/side-write")


async def test_and_work_through_app_state_is_rolled_back_too(db_session: AsyncSession) -> None:
    assert await db_session.scalar(text("SELECT to_regclass('side_effect')")) is None


async def test_test_sessions_roll_back_even_committed_work(db_session: AsyncSession) -> None:
    await db_session.execute(text("CREATE TABLE scratch (id int)"))
    await db_session.commit()
    assert await db_session.scalar(text("SELECT to_regclass('scratch')")) == "scratch"


async def test_and_the_next_test_does_not_see_it(db_session: AsyncSession) -> None:
    assert await db_session.scalar(text("SELECT to_regclass('scratch')")) is None


async def test_ready_checks_the_real_database(database_url: str) -> None:
    app = create_app(Settings.model_validate({"app_env": "test", "database_url": database_url}))
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
    assert response.status_code == 200


async def test_ready_fails_when_the_database_is_unreachable() -> None:
    url = "postgresql+psycopg://app:app@127.0.0.1:1/app?connect_timeout=1"
    app = create_app(Settings.model_validate({"app_env": "test", "database_url": url}))
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["failing"] == ["database"]


async def test_ping_gives_up_instead_of_hanging() -> None:
    # 10.255.255.1 is unroutable: without a bound the connect would wait for the TCP timeout.
    engine = create_engine(
        Settings.model_validate({"database_url": "postgresql+psycopg://app:app@10.255.255.1/app"})
    )
    with pytest.raises(TimeoutError):
        await ping(engine, seconds=0.5)
    await engine.dispose()


def test_a_template_update_never_collides_with_project_migrations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The template's migrations and the project's grow on separate branches, so a template
    # update that ships a migration leaves one head per branch, and `upgrade heads` applies both.
    scripts = tmp_path / "migrations"
    shutil.copytree(db.BACKEND / "migrations", scripts)
    config = db.alembic_config("postgresql+psycopg://app:app@localhost/unused")
    config.set_main_option("script_location", str(scripts))
    config.set_section_option("post_write_hooks", "hooks", "")  # no formatting needed here
    # Whatever each branch's head is now: the project's own migrations and later template ones
    # move them (Codex review on PR 19).
    before = ScriptDirectory.from_config(config).revision_map
    app_head = before.get_current_head("app")
    template_head = before.get_current_head("template")
    command.revision(config, "a project table", head="app@head", rev_id="p1")
    command.revision(config, "a later template change", head="template@head", rev_id="t9")

    heads = ScriptDirectory.from_config(config)
    assert sorted(heads.get_heads()) == ["p1", "t9"]
    # Each new revision sits on its own branch's line.
    assert heads.get_revision("p1").down_revision == app_head
    assert heads.get_revision("t9").down_revision == template_head
    command.upgrade(config, "heads", sql=True)  # offline: writes the SQL, no database needed
    sql = capsys.readouterr().out
    assert "INSERT INTO alembic_version" in sql


def test_only_the_project_root_starts_the_app_branch() -> None:
    script = ScriptDirectory.from_config(db.alembic_config("postgresql+psycopg://unused/x"))
    roots = {
        rev.revision: rev.branch_labels for rev in script.walk_revisions() if not rev.down_revision
    }
    assert roots == {"0001": {"template"}, "app_root": {"app"}}
