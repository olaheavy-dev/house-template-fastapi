"""Model-call failures, whatever framework made the call: what kind of failure it was, whether the
provider billed it, and what the client is told.

An adapter classifies its framework's exceptions as a FailureKind (app/llm/langchain/errors.py for
LangChain) or raises ProviderError itself. The kind decides two things, each in one table here:
OUTCOMES, the problem response (status, detail, whether to send Retry-After), and REJECTED, the
kinds a provider answers before doing any work, so the call's reservation is given back.

A kind is never stored: every failed call is recorded as status "failed" in llm_call and counted
with outcome "failed" in llm.calls, whatever its kind. Nothing from the provider's message reaches
the client or this module's types: it can echo the prompt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.problems import problem_response

__all__ = [
    "DEFAULT_RETRY_AFTER",
    "OUTCOMES",
    "REJECTED",
    "FailureKind",
    "Outcome",
    "ProviderError",
    "StepLimitReached",
    "failure_response",
    "install_failure_handlers",
    "outcome",
    "step_limit_response",
]

log = structlog.get_logger()
# When the provider doesn't say how long to wait.
DEFAULT_RETRY_AFTER = "30"


class FailureKind(StrEnum):
    """Why a model call failed, in the provider-independent terms of an HTTP status."""

    RATE_LIMITED = "rate_limited"  # 429
    TIMED_OUT = "timed_out"
    CONTEXT_OVERFLOW = "context_overflow"  # the input is longer than the model's context
    UNREACHABLE = "unreachable"  # no connection to the provider
    PROVIDER_FAILED = "provider_failed"  # 5xx, or any other provider error
    AUTHENTICATION = "authentication"  # 401
    PERMISSION_DENIED = "permission_denied"  # 403
    MODEL_NOT_FOUND = "model_not_found"  # 404
    # 400: the request the app built is wrong, a bug. It has no outcome: a plain 500.
    INVALID_REQUEST = "invalid_request"


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int
    detail: str
    retry: bool = False


_UNAVAILABLE = Outcome(503, "The language model isn't available.")
OUTCOMES: Mapping[FailureKind, Outcome] = MappingProxyType(
    {
        FailureKind.RATE_LIMITED: Outcome(
            503, "The language model is busy. Try again shortly.", True
        ),
        FailureKind.TIMED_OUT: Outcome(504, "The language model took too long to answer."),
        FailureKind.CONTEXT_OVERFLOW: Outcome(422, "The input is too long for the language model."),
        FailureKind.UNREACHABLE: Outcome(502, "The language model can't be reached.", True),
        FailureKind.PROVIDER_FAILED: Outcome(502, "The language model failed to answer.", True),
        # The app's own configuration is wrong (key, permission, model id): not the caller's fault.
        FailureKind.AUTHENTICATION: _UNAVAILABLE,
        FailureKind.PERMISSION_DENIED: _UNAVAILABLE,
        FailureKind.MODEL_NOT_FOUND: _UNAVAILABLE,
    }
)
# The provider refused these before doing any work, so nothing was billed: the call settles at 0
# (unless it had already streamed tokens). Any other failure, or one of no known kind, may have
# been billed and keeps its reservation.
REJECTED: frozenset[FailureKind] = frozenset(
    {
        FailureKind.AUTHENTICATION,
        FailureKind.PERMISSION_DENIED,
        FailureKind.INVALID_REQUEST,
        FailureKind.MODEL_NOT_FOUND,
        FailureKind.RATE_LIMITED,
        FailureKind.CONTEXT_OVERFLOW,
    }
)


def outcome(kind: FailureKind | None) -> Outcome | None:
    """The answer for a kind; None (a plain 500) for an invalid request or an unknown failure."""
    return None if kind is None else OUTCOMES.get(kind)


class ProviderError(Exception):
    """A failed model call as an adapter reports it, carrying no provider text. `source` names the
    error the adapter saw (its type name, e.g. "RateLimitError"), for the log; `retry_after` is the
    provider's Retry-After value, sent on only when the kind's outcome says retry. The status is the
    kind's (OUTCOMES), never the adapter's. Raise it `from` the provider's error."""

    def __init__(self, kind: FailureKind, *, source: str, retry_after: str | None = None) -> None:
        super().__init__(f"{source}: {kind.value}")
        self.kind = kind
        self.source = source
        self.retry_after = retry_after


class StepLimitReached(Exception):
    """An agent stopped at its step limit (AGENT_MAX_STEPS) before it finished."""


def failure_response(
    request: Request,
    kind: FailureKind | None,
    *,
    source: str,
    retry_after: str | None,
    error: BaseException,
) -> JSONResponse:
    """The problem response for a failed model call, logged as llm_error. `error` is what was
    raised; its traceback is logged when the failure is a bug (no outcome)."""
    path = request.url.path
    answer = outcome(kind)
    if answer is None:  # an invalid request the app built: a bug, handled like any other
        log.error("llm_error", kind=source, path=path, exc_info=error)
        return problem_response(500, instance=path)
    log.warning("llm_error", kind=source, status=answer.status, path=path)
    headers = {"Retry-After": retry_after or DEFAULT_RETRY_AFTER} if answer.retry else {}
    return problem_response(answer.status, detail=answer.detail, instance=path, headers=headers)


def step_limit_response(request: Request) -> JSONResponse:
    log.warning("agent_step_limit", path=request.url.path)
    return problem_response(
        422,
        detail="The assistant couldn't finish this within its step limit.",
        instance=request.url.path,
    )


async def _provider_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)  # noqa: S101 - registered for ProviderError only
    return failure_response(
        request, exc.kind, source=exc.source, retry_after=exc.retry_after, error=exc
    )


async def _step_limit(request: Request, exc: Exception) -> JSONResponse:
    return step_limit_response(request)


def install_failure_handlers(app: FastAPI) -> None:
    """Answers ProviderError and StepLimitReached, whichever adapter raised them. Each adapter adds
    its own framework's exceptions (app.llm.langchain.errors.install_model_error_handlers)."""
    app.add_exception_handler(ProviderError, _provider_error)
    app.add_exception_handler(StepLimitReached, _step_limit)
