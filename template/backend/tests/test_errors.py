"""Failures as the client sees them, and what the logs keep of them."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from structlog.testing import capture_logs

from app.log import configure_logging
from app.problems import Problem
from app.settings import Settings

pytestmark = pytest.mark.anyio


async def test_a_problem_carries_its_headers(app: FastAPI, client: httpx.AsyncClient) -> None:
    @app.get("/busy")
    async def busy() -> None:
        raise Problem(503, detail="Busy.", headers={"Retry-After": "5"})

    response = await client.get("/busy")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


def test_extensions_cannot_overwrite_the_standard_members() -> None:
    with pytest.raises(ValueError, match="instance"):
        Problem(400, instance="/elsewhere")


async def test_a_structured_http_exception_detail_becomes_errors(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/structured")
    async def structured() -> None:
        raise HTTPException(409, detail={"field": "taken"})

    body = (await client.get("/structured")).json()
    assert "detail" not in body
    assert body["errors"] == {"field": "taken"}


async def test_each_request_logs_one_access_line_with_the_route_template(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/courses/{course_id}")
    async def course(course_id: int) -> int:
        return course_id

    with capture_logs() as logs:
        await client.get("/courses/42")
    [line] = [log for log in logs if log["event"] == "request"]
    assert line["route"] == "/courses/{course_id}"  # never the id itself
    assert (line["method"], line["status"]) == ("GET", 200)
    assert line["duration_ms"] >= 0


def test_library_logs_are_json_and_the_duplicate_traceback_is_dropped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(Settings(_env_file=None, app_env="test", log_json=True))
    logging.getLogger("httpx").warning("pool exhausted")
    logging.getLogger("uvicorn.error").error("Exception in ASGI application\n")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET / HTTP/1.1" 200')
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert (event["event"], event["level"]) == ("pool exhausted", "warning")
    assert "timestamp" in event
