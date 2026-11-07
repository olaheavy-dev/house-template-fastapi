from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.health import add_readiness_check

pytestmark = pytest.mark.anyio


async def test_live_answers_without_touching_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_passes_when_every_check_passes(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    async def fine() -> None:
        return None

    add_readiness_check(app, "database", fine)
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_names_the_failing_check_and_returns_503(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    async def fine() -> None:
        return None

    async def down() -> None:
        raise ConnectionError("database unreachable")

    add_readiness_check(app, "database", down)
    add_readiness_check(app, "cache", fine)
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "failing": ["database"]}


async def test_ready_is_ok_when_no_check_is_registered(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    # A project without a database or storage registers nothing, and is ready once it is up.
    app.state.readiness_checks.clear()
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_checks_only_what_the_settings_turn_on(
    app: FastAPI, client: httpx.AsyncClient, expected_readiness_checks: set[str]
) -> None:
    # Without a storage bucket, only the parts that always check (the database, when the project
    # has one) are registered.
    assert set(app.state.readiness_checks) == expected_readiness_checks
