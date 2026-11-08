"""How model failures and an exhausted budget reach the client: problem responses that never carry
the provider's message."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
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

from app.llm.core.budget import BudgetExhausted

pytestmark = pytest.mark.anyio


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
