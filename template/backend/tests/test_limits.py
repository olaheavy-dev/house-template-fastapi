from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.anyio


class Note(BaseModel):
    text: str


@pytest.fixture
def small_app() -> FastAPI:
    app = create_app(Settings(_env_file=None, app_env="test", max_request_bytes=10))

    @app.post("/echo")
    async def echo(request: Request) -> int:
        return len(await request.body())

    @app.post("/notes")
    async def notes(note: Note) -> str:
        return note.text

    return app


async def post(app: FastAPI, path: str = "/echo", **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, **kwargs)  # type: ignore[arg-type]


async def test_a_body_within_the_limit_is_read(small_app: FastAPI) -> None:
    response = await post(small_app, content=b"0123456789")
    assert (response.status_code, response.json()) == (200, 10)


async def test_a_declared_oversized_body_is_refused_before_it_is_read(small_app: FastAPI) -> None:
    response = await post(small_app, content=b"x" * 11)
    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
    assert "X-Request-ID" in response.headers  # request_context wraps the limit


async def test_a_streamed_body_without_a_length_stops_at_the_limit(small_app: FastAPI) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(5):
            yield b"abcd"

    response = await post(small_app, content=chunks())
    assert response.status_code == 413


async def test_a_streamed_body_to_a_typed_endpoint_is_a_413_not_a_parse_error(
    small_app: FastAPI,
) -> None:
    # Codex review on PR 22: FastAPI parses typed bodies itself and turns other exceptions raised
    # while reading into a 400.
    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"text": "'
        yield b"0123456789"
        yield b'"}'

    response = await post(
        small_app, "/notes", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
