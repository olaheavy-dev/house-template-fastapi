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
    add_readiness_check(app, "storage", fine)
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "failing": ["database"]}
