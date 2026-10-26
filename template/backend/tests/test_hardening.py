"""Production defaults: settings that refuse to start wrong, database limits, headers, CORS and
allowed hosts."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Response
from pydantic import SecretStr, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.db import create_engine
from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.anyio

PRODUCTION_DB = "postgresql+psycopg://app:secret@db.internal:5432/app"


def test_the_tests_never_read_a_developers_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("LOG_LEVEL=DEBUG\nOPENAI_API_KEY=sk-from-a-real-env-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings()
    assert settings.log_level == "INFO"
    assert settings.openai_api_key is None


def test_production_refuses_the_local_database_and_storage_defaults() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL is the local default"):
        Settings(_env_file=None, app_env="production")
    with pytest.raises(ValidationError, match="STORAGE_SECRET_ACCESS_KEY is the local default"):
        Settings(
            _env_file=None, app_env="production", database_url=PRODUCTION_DB, storage_bucket="b"
        )
    ok = Settings(
        _env_file=None,
        app_env="production",
        database_url=PRODUCTION_DB,
        storage_bucket="b",
        storage_secret_access_key=SecretStr("a-real-secret"),
    )
    assert ok.app_env == "production"


async def test_a_runaway_query_is_cancelled_by_the_statement_timeout(database_url: str) -> None:
    engine = create_engine(
        Settings(_env_file=None, database_url=database_url, db_statement_timeout_ms=200)
    )
    try:
        async with engine.connect() as connection:
            with pytest.raises(DBAPIError, match="statement timeout"):
                await connection.execute(text("SELECT pg_sleep(2)"))
    finally:
        await engine.dispose()


def add_routes(app: FastAPI) -> None:
    @app.get("/plain")
    async def plain() -> str:
        return "ok"

    @app.get("/cached")
    async def cached(response: Response) -> str:
        response.headers["Cache-Control"] = "public, max-age=60"
        return "ok"


async def test_responses_carry_the_baseline_security_headers(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    add_routes(app)
    plain = await client.get("/plain")
    assert plain.headers["x-content-type-options"] == "nosniff"
    assert plain.headers["x-frame-options"] == "DENY"
    assert plain.headers["referrer-policy"] == "no-referrer"
    assert plain.headers["cache-control"] == "no-store"
    # A route that sets its own caching keeps it; errors get the headers too.
    assert (await client.get("/cached")).headers["cache-control"] == "public, max-age=60"
    assert (await client.get("/nowhere")).headers["x-content-type-options"] == "nosniff"


async def request(app: FastAPI, method: str, path: str, **headers: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, headers=headers)


async def test_cors_is_off_unless_origins_are_listed() -> None:
    app = create_app(Settings(_env_file=None, app_env="test"))
    add_routes(app)
    response = await request(app, "GET", "/plain", Origin="https://app.example.com")
    assert "access-control-allow-origin" not in response.headers


async def test_listed_origins_may_call_the_api_with_credentials() -> None:
    app = create_app(
        Settings(_env_file=None, app_env="test", cors_origins=["https://app.example.com"])
    )
    add_routes(app)
    preflight = await request(
        app,
        "OPTIONS",
        "/plain",
        Origin="https://app.example.com",
        **{"Access-Control-Request-Method": "GET"},
    )
    assert preflight.headers["access-control-allow-origin"] == "https://app.example.com"
    assert preflight.headers["access-control-allow-credentials"] == "true"
    other = await request(app, "GET", "/plain", Origin="https://evil.example.com")
    assert "access-control-allow-origin" not in other.headers


async def test_only_allowed_hosts_are_answered() -> None:
    app = create_app(Settings(_env_file=None, app_env="test", allowed_hosts=["api.example.com"]))
    add_routes(app)
    assert (await request(app, "GET", "/plain")).status_code == 400  # Host: test
    assert (await request(app, "GET", "/plain", Host="api.example.com")).status_code == 200


async def test_the_budget_has_its_own_pool(app: FastAPI) -> None:
    # A request holding a session from the main pool must never wait on that pool to reserve.
    async with LifespanManager(app):
        budget = app.state.budget_sessionmaker.kw["bind"]
        assert budget is not app.state.engine
        assert budget.pool.size() == 2


async def test_an_unexpected_error_still_carries_cors_and_the_headers() -> None:
    # Codex review on PR 23: Starlette's catch-all 500 handler runs outside the app's middleware.
    app = create_app(
        Settings(_env_file=None, app_env="test", cors_origins=["https://app.example.com"])
    )

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("database password is hunter2")

    response = await request(app, "GET", "/boom", Origin="https://app.example.com")
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert len(response.headers["x-request-id"]) == 32
    assert "hunter2" not in response.text
