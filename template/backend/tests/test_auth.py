from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.auth import DEV_USER_HEADER, CurrentUser, OptionalUser
from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.anyio


def add_routes(app: FastAPI) -> None:
    @app.get("/me")
    async def me(user: CurrentUser) -> str:
        return user.id

    @app.get("/maybe")
    async def maybe(user: OptionalUser) -> str | None:
        return user.id if user else None


async def test_a_protected_route_needs_a_user(app: FastAPI, client: httpx.AsyncClient) -> None:
    add_routes(app)
    response = await client.get("/me")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["content-type"] == "application/problem+json"


async def test_the_dev_header_stands_in_for_a_user_outside_production(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    add_routes(app)
    assert (await client.get("/me", headers={DEV_USER_HEADER: "user-1"})).json() == "user-1"
    assert (await client.get("/maybe")).json() is None


async def test_production_fails_closed_until_real_authentication_exists() -> None:
    app = create_app(
        Settings(
            _env_file=None,
            app_env="production",
            database_url="postgresql+psycopg://app:secret@db.internal:5432/app",
        )
    )
    add_routes(app)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/me", headers={DEV_USER_HEADER: "admin"})
    assert response.status_code == 401


@pytest.mark.parametrize("claimed", ["", "has space", "x" * 201, "line\nbreak"])
async def test_unsafe_dev_ids_are_refused(
    app: FastAPI, client: httpx.AsyncClient, claimed: str
) -> None:
    add_routes(app)
    assert (await client.get("/me", headers={DEV_USER_HEADER: claimed})).status_code == 401
