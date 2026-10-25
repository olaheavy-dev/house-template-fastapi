from __future__ import annotations

import httpx
import pytest
from alembic import command
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionDep
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
    command.upgrade(config, "head")


async def test_requests_get_a_session(app: FastAPI, db_client: httpx.AsyncClient) -> None:
    @app.get("/one")
    async def one(session: SessionDep) -> int:
        return int(await session.scalar(text("SELECT 1")) or 0)

    assert (await db_client.get("/one")).json() == 1


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
