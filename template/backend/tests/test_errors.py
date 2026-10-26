"""Failures as the client sees them, and what the logs keep of them."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from structlog.testing import capture_logs

from app.llm.budget import BudgetExhausted
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


@pytest.mark.parametrize(
    ("error", "status", "retry_after"),
    [
        (ModelRateLimitError("slow down: prompt was SECRET"), 503, "30"),
        (ModelTimeoutError("timed out"), 504, None),
        (ContextOverflowError("context: 900k tokens"), 422, None),
        (ModelConnectionError("reset"), 502, "30"),
        (ModelAPIError("500 from provider"), 502, "30"),
        (ModelAuthenticationError("bad key"), 503, None),
    ],
)
async def test_model_failures_become_problems_without_the_providers_message(
    app: FastAPI, client: httpx.AsyncClient, error: ModelError, status: int, retry_after: str | None
) -> None:
    @app.get("/model")
    async def model() -> None:
        raise error

    response = await client.get("/model")
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers.get("Retry-After") == retry_after
    assert "SECRET" not in response.text
    assert str(error) not in response.text


async def test_the_providers_retry_after_is_passed_on(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    limited = ModelRateLimitError("slow down")
    limited.response = SimpleNamespace(headers={"retry-after": "7"})  # type: ignore[attr-defined]

    @app.get("/limited")
    async def limited_route() -> None:
        raise limited

    assert (await client.get("/limited")).headers["Retry-After"] == "7"


async def test_an_invalid_request_the_app_built_is_a_plain_500(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/bug")
    async def bug() -> None:
        raise ModelInvalidRequestError("unknown parameter")

    response = await client.get("/bug")
    assert response.status_code == 500
    assert "unknown parameter" not in response.text


def test_the_budget_says_to_retry_at_midnight_utc() -> None:
    refused = BudgetExhausted(now=datetime(2026, 1, 2, 23, 59, 30, tzinfo=UTC))
    assert (refused.status, refused.headers["Retry-After"]) == (503, "30")


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
    logging.getLogger("sqlalchemy.engine").warning("pool exhausted")
    logging.getLogger("uvicorn.error").error("Exception in ASGI application\n")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET / HTTP/1.1" 200')
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert (event["event"], event["level"]) == ("pool exhausted", "warning")
    assert "timestamp" in event
