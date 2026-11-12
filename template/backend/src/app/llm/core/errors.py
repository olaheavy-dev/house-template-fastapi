"""Model-call failures, whatever framework made the call: what kind of failure it was, whether the
provider billed it, and what the client is told.

An adapter classifies its framework's exceptions as a FailureKind (app/llm/langchain/errors.py for
LangChain) or raises ProviderError itself. The kind decides two things, each in one table here:
OUTCOMES, the problem response (status, detail, whether to send Retry-After), and REJECTED, the
kinds a provider answers before doing any work, so the call's reservation is given back.

A kind is never stored: every failed call is recorded as status "failed" in llm_call and counted
with outcome "failed" in llm.calls, whatever its kind. Nothing from the provider's message reaches
the client or this module's types: it can echo the prompt.

A structured answer that does not match the schema the call asked for is a SchemaMismatch, a 502
(the founder's decision on #35, 11 Nov 2026): the call was made and is settled at its cost, and
neither the answer nor the parser's message (both hold the model's text) reaches the client or the
log. OpenAI and Anthropic both say a refusal or an answer cut off at the output limit can break the
schema.
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath
from types import MappingProxyType

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.problems import Problem, problem_response

__all__ = [
    "DEFAULT_RETRY_AFTER",
    "OUTCOMES",
    "REJECTED",
    "SCHEMA_MISMATCH_DETAIL",
    "FailureKind",
    "Outcome",
    "ProviderError",
    "SchemaMismatch",
    "failure_response",
    "install_failure_handlers",
    "outcome",
    "schema_mismatch",
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


def _chain(error: BaseException) -> list[BaseException]:
    links: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(links) < 16:
        links.append(current)
        current = current.__cause__ or current.__context__
    return links


def _type_names(error: BaseException) -> list[str]:
    return [type(link).__name__ for link in _chain(error)]


def _where(error: BaseException) -> list[str]:
    """Where each error in the chain was raised, as `file.py:123` (the innermost frame of its
    traceback): enough to find the bug, with none of the messages."""
    found: list[str] = []
    for link in _chain(error):
        frames = traceback.extract_tb(link.__traceback__)
        if frames:
            found.append(f"{PurePath(frames[-1].filename).name}:{frames[-1].lineno}")
    return found


SCHEMA_MISMATCH_DETAIL = "The language model's answer didn't match the expected format."


class SchemaMismatch(Problem):
    """A structured answer that did not match its schema (502; see the module notes). It carries
    no text and no cause: raise it `from None`, and log only the parser error's type name
    (schema_mismatch does both)."""

    def __init__(self) -> None:
        super().__init__(502, detail=SCHEMA_MISMATCH_DETAIL)


def schema_mismatch(error: BaseException) -> SchemaMismatch:
    """The SchemaMismatch for a parser's `error`, logged by its type name only."""
    log.warning("llm_schema_mismatch", error_type=type(error).__name__)
    return SchemaMismatch()


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
        # The type names of the error and of what it was raised from, never a traceback: a
        # framework's wrapper (MAF's) embeds the provider's text, which can echo the prompt, in the
        # messages of the chain a traceback prints.
        log.error(
            "llm_error", kind=source, path=path, chain=_type_names(error), where=_where(error)
        )
        return problem_response(500, instance=path)
    log.warning("llm_error", kind=source, status=answer.status, path=path)
    headers = {"Retry-After": retry_after or DEFAULT_RETRY_AFTER} if answer.retry else {}
    return problem_response(answer.status, detail=answer.detail, instance=path, headers=headers)


async def _provider_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProviderError)  # noqa: S101 - registered for ProviderError only
    return failure_response(
        request, exc.kind, source=exc.source, retry_after=exc.retry_after, error=exc
    )


def install_failure_handlers(app: FastAPI) -> None:
    """Answers ProviderError, whichever adapter raised it. Each adapter adds its own framework's
    exceptions (app.llm.langchain.errors.install_model_error_handlers)."""
    app.add_exception_handler(ProviderError, _provider_error)
