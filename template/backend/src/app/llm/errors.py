"""Model provider failures as problem responses instead of bare 500s.

LangChain raises one family of exceptions for every provider (langchain_core.exceptions, keyed to
the HTTP status the provider returned), so one handler covers OpenAI and Anthropic alike. The
client learns what to do (wait, retry later, send less); the log says which provider error it was.
Nothing from the provider's message reaches the client: it can echo the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.errors import GraphRecursionError

from app.problems import problem_response

log = structlog.get_logger()
# When the provider doesn't say how long to wait.
DEFAULT_RETRY_AFTER = "30"


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int
    detail: str
    retry: bool = False


# Most specific first: the first matching class wins.
OUTCOMES: tuple[tuple[type[ModelError], Outcome], ...] = (
    (ModelRateLimitError, Outcome(503, "The language model is busy. Try again shortly.", True)),
    (ModelTimeoutError, Outcome(504, "The language model took too long to answer.")),
    (ContextOverflowError, Outcome(422, "The input is too long for the language model.")),
    (ModelConnectionError, Outcome(502, "The language model can't be reached.", True)),
    (ModelAPIError, Outcome(502, "The language model failed to answer.", True)),
    # The app's own configuration is wrong (key, permission, model id): not the caller's fault.
    (ModelAuthenticationError, Outcome(503, "The language model isn't available.")),
    (ModelPermissionDeniedError, Outcome(503, "The language model isn't available.")),
    (ModelNotFoundError, Outcome(503, "The language model isn't available.")),
)


def outcome_for(exc: ModelError) -> Outcome | None:
    return next((outcome for kind, outcome in OUTCOMES if isinstance(exc, kind)), None)


async def _model_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ModelError)  # noqa: S101 - registered for ModelError only
    outcome = outcome_for(exc)
    if outcome is None:  # an invalid request the app built: a bug, handled like any other
        log.error("llm_error", kind=type(exc).__name__, path=request.url.path, exc_info=exc)
        return problem_response(500, instance=request.url.path)
    log.warning("llm_error", kind=type(exc).__name__, status=outcome.status, path=request.url.path)
    headers = {}
    if outcome.retry:
        provider = getattr(getattr(exc, "response", None), "headers", None) or {}
        headers["Retry-After"] = str(provider.get("retry-after") or DEFAULT_RETRY_AFTER)
    return problem_response(
        outcome.status, detail=outcome.detail, instance=request.url.path, headers=headers
    )


async def _step_limit(request: Request, exc: Exception) -> JSONResponse:
    log.warning("agent_step_limit", path=request.url.path)
    return problem_response(
        422,
        detail="The assistant couldn't finish this within its step limit.",
        instance=request.url.path,
    )


def install_model_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ModelError, _model_error)
    app.add_exception_handler(GraphRecursionError, _step_limit)
