"""Characterisation of what a model failure logs (#33 T2): the event names and fields dashboards and
alerts read, pinned before the errors move to the framework-neutral core. Written against the
LangChain exceptions only, so it runs unchanged before and after the move."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.exceptions import (
    ModelError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.errors import GraphRecursionError
from structlog.testing import capture_logs

pytestmark = pytest.mark.anyio

EVENTS = {"llm_error", "agent_step_limit"}


async def logged(
    app: FastAPI, client: httpx.AsyncClient, error: Exception
) -> list[MutableMapping[str, Any]]:
    @app.get("/fail")
    async def fail() -> None:
        raise error

    with capture_logs() as logs:
        await client.get("/fail")
    return [entry for entry in logs if entry["event"] in EVENTS]


class ProviderRateLimit(ModelRateLimitError):
    """A provider integration's own class, as langchain-openai and langchain-anthropic raise."""


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [
        (ModelRateLimitError("SECRET"), "ModelRateLimitError", 503),
        (ModelTimeoutError("SECRET"), "ModelTimeoutError", 504),
        # The raised class's own name, not the family class it matched.
        (ProviderRateLimit("SECRET"), "ProviderRateLimit", 503),
    ],
)
async def test_a_mapped_failure_logs_one_warning_with_its_class_status_and_path(
    app: FastAPI, client: httpx.AsyncClient, error: Exception, kind: str, status: int
) -> None:
    assert await logged(app, client, error) == [
        {
            "event": "llm_error",
            "log_level": "warning",
            "kind": kind,
            "status": status,
            "path": "/fail",
        }
    ]


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ModelInvalidRequestError("SECRET"), "ModelInvalidRequestError"),
        (ModelError("x"), "ModelError"),
    ],
)
async def test_an_unmapped_failure_logs_an_error_with_the_exception_and_no_status(
    app: FastAPI, client: httpx.AsyncClient, error: Exception, kind: str
) -> None:
    # exc_info is the exception the route raised (structlog renders its traceback).
    assert await logged(app, client, error) == [
        {
            "event": "llm_error",
            "log_level": "error",
            "kind": kind,
            "path": "/fail",
            "exc_info": error,
        }
    ]


async def test_the_agent_step_limit_logs_a_warning_with_the_path(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    assert await logged(app, client, GraphRecursionError("recursion")) == [
        {"event": "agent_step_limit", "log_level": "warning", "path": "/fail"}
    ]
