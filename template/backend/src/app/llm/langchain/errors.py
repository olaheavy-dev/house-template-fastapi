"""LangChain's model errors as the core's failure kinds (app/llm/core/errors.py).

LangChain raises one family of exceptions for every provider (langchain_core.exceptions, keyed to
the HTTP status the provider returned), so one table covers OpenAI and Anthropic alike. The
exceptions reach FastAPI as LangChain raised them, so the handlers here classify each one and answer
it with the core's response; the budget classifies the same way to decide what a failed call cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.errors import GraphRecursionError

from app.llm.core import errors as core
from app.llm.core.errors import FailureKind, Outcome

__all__ = [
    "KINDS",
    "OUTCOMES",
    "install_model_error_handlers",
    "kind_of",
    "model_error",
    "outcome_for",
    "retry_after_of",
]

# Most specific first: the first matching class wins (a provider's class may inherit from two).
KINDS: tuple[tuple[type[ModelError], FailureKind], ...] = (
    (ModelRateLimitError, FailureKind.RATE_LIMITED),
    (ModelTimeoutError, FailureKind.TIMED_OUT),
    (ContextOverflowError, FailureKind.CONTEXT_OVERFLOW),
    (ModelConnectionError, FailureKind.UNREACHABLE),
    (ModelAPIError, FailureKind.PROVIDER_FAILED),
    (ModelAuthenticationError, FailureKind.AUTHENTICATION),
    (ModelPermissionDeniedError, FailureKind.PERMISSION_DENIED),
    (ModelNotFoundError, FailureKind.MODEL_NOT_FOUND),
    (ModelInvalidRequestError, FailureKind.INVALID_REQUEST),
)
# Each class's answer, in KINDS order; ModelInvalidRequestError has none (a plain 500).
OUTCOMES: tuple[tuple[type[ModelError], Outcome], ...] = tuple(
    (cls, core.OUTCOMES[kind]) for cls, kind in KINDS if kind in core.OUTCOMES
)
# The LangChain class for each kind, for code that turns a provider SDK's error into LangChain's.
_ERRORS: Mapping[FailureKind, type[ModelError]] = MappingProxyType(
    {kind: cls for cls, kind in KINDS}
)


def kind_of(error: BaseException) -> FailureKind | None:
    """The kind of a LangChain model error; None for anything else, plain ModelError included."""
    return next((kind for cls, kind in KINDS if isinstance(error, cls)), None)


def outcome_for(exc: ModelError) -> Outcome | None:
    return core.outcome(kind_of(exc))


def retry_after_of(error: BaseException) -> str | None:
    """The provider's Retry-After header, when the error carries the provider's response."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    value = headers.get("retry-after")
    return str(value) if value else None


def model_error(kind: FailureKind, message: str) -> ModelError:
    """LangChain's error for a kind. The message must not hold the provider's text."""
    return _ERRORS[kind](message)


async def _model_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ModelError)  # noqa: S101 - registered for ModelError only
    return core.failure_response(
        request,
        kind_of(exc),
        source=type(exc).__name__,
        retry_after=retry_after_of(exc),
        error=exc,
    )


async def _step_limit(request: Request, exc: Exception) -> JSONResponse:
    return core.step_limit_response(request)


def install_model_error_handlers(app: FastAPI) -> None:
    """Answers LangChain's model errors and LangGraph's recursion limit (an agent's step limit).
    The core's own handlers are installed apart (app.llm.core.errors.install_failure_handlers)."""
    app.add_exception_handler(ModelError, _model_error)
    app.add_exception_handler(GraphRecursionError, _step_limit)
