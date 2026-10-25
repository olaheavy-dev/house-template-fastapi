from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from app.problems import Problem

pytestmark = pytest.mark.anyio


class Course(BaseModel):
    title: str
    weeks: int


@pytest.fixture
def app(app: FastAPI) -> FastAPI:
    @app.get("/courses/{course_id}")
    async def get_course(course_id: int) -> Course:
        raise Problem(404, detail=f"No course {course_id}.")

    @app.post("/courses")
    async def create_course(course: Course) -> Course:
        return course

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    return app


async def test_problem_becomes_rfc9457_json(client: httpx.AsyncClient) -> None:
    response = await client.get("/courses/7")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "No course 7.",
        "instance": "/courses/7",
    }


async def test_unknown_route_is_a_problem_too(client: httpx.AsyncClient) -> None:
    response = await client.get("/nowhere")
    assert response.status_code == 404
    assert response.json()["title"] == "Not Found"


async def test_validation_error_lists_fields_but_never_echoes_input(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/courses", json={"title": "Private note", "weeks": "many"})
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == 422
    assert body["errors"][0]["loc"] == ["body", "weeks"]
    assert "many" not in response.text
    assert "Private note" not in response.text


async def test_unexpected_error_hides_details(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")
    assert response.status_code == 500
    assert response.json() == {
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": 500,
        "instance": "/boom",
    }
    assert "secret" not in response.text


async def test_unexpected_error_still_carries_the_request_id(app: FastAPI) -> None:
    # The 500 handler runs outside the request-id middleware, so it has to add the header itself.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom", headers={"X-Request-ID": "req-42"})
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-42"
