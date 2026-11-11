"""The framework-neutral failure family (#33 T2): the kinds, what each one answers and whether the
provider billed it, ProviderError's problem responses and logs, and that none of it needs a model
framework. Expected values are written out by hand, never read from the tables under test."""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest
from fastapi import FastAPI
from structlog.testing import capture_logs

from app.llm.core.errors import (
    DEFAULT_RETRY_AFTER,
    OUTCOMES,
    REJECTED,
    FailureKind,
    Outcome,
    ProviderError,
    outcome,
)
from tests.test_llm_core import FRAMEWORKS

pytestmark = pytest.mark.anyio

K = FailureKind
BUSY = "The language model is busy. Try again shortly."
UNAVAILABLE = "The language model isn't available."
# kind: (status, detail, Retry-After sent), as test_llm_surface.py pins them per LangChain class.
ANSWERS = {
    K.RATE_LIMITED: (503, BUSY, True),
    K.TIMED_OUT: (504, "The language model took too long to answer.", False),
    K.CONTEXT_OVERFLOW: (422, "The input is too long for the language model.", False),
    K.UNREACHABLE: (502, "The language model can't be reached.", True),
    K.PROVIDER_FAILED: (502, "The language model failed to answer.", True),
    K.AUTHENTICATION: (503, UNAVAILABLE, False),
    K.PERMISSION_DENIED: (503, UNAVAILABLE, False),
    K.MODEL_NOT_FOUND: (503, UNAVAILABLE, False),
}


def test_the_kinds_and_their_values() -> None:
    assert {kind.name: kind.value for kind in FailureKind} == {
        "RATE_LIMITED": "rate_limited",
        "TIMED_OUT": "timed_out",
        "CONTEXT_OVERFLOW": "context_overflow",
        "UNREACHABLE": "unreachable",
        "PROVIDER_FAILED": "provider_failed",
        "AUTHENTICATION": "authentication",
        "PERMISSION_DENIED": "permission_denied",
        "MODEL_NOT_FOUND": "model_not_found",
        "INVALID_REQUEST": "invalid_request",
    }
    # A kind is never a call's status or a metric outcome: those stay reserved/settled/failed/
    # refused, and every kind is recorded as "failed".
    assert {k.value for k in FailureKind}.isdisjoint({"reserved", "settled", "failed", "refused"})


def test_every_kind_but_an_invalid_request_has_this_outcome() -> None:
    assert {k: (o.status, o.detail, o.retry) for k, o in OUTCOMES.items()} == ANSWERS
    assert outcome(K.INVALID_REQUEST) is None
    assert outcome(None) is None
    assert outcome(K.TIMED_OUT) == Outcome(504, "The language model took too long to answer.")
    assert DEFAULT_RETRY_AFTER == "30"
    with pytest.raises(TypeError):
        OUTCOMES[K.INVALID_REQUEST] = Outcome(400, "x")  # type: ignore[index]


def test_the_rejected_kinds_are_the_ones_answered_before_any_work() -> None:
    # 401, 403, 400, 404, 429 and context overflow: what budget.py and embeddings.py released.
    assert {
        K.AUTHENTICATION,
        K.PERMISSION_DENIED,
        K.INVALID_REQUEST,
        K.MODEL_NOT_FOUND,
        K.RATE_LIMITED,
        K.CONTEXT_OVERFLOW,
    } == REJECTED
    # May have been billed: a timeout, a dropped connection, a 5xx.
    assert set(FailureKind) - REJECTED == {K.TIMED_OUT, K.UNREACHABLE, K.PROVIDER_FAILED}


def test_a_provider_error_carries_its_kind_source_and_retry_after_and_no_provider_text() -> None:
    error = ProviderError(K.RATE_LIMITED, source="RateLimitError", retry_after="7")
    assert (error.kind, error.source, error.retry_after) == (K.RATE_LIMITED, "RateLimitError", "7")
    assert str(error) == "RateLimitError: rate_limited"
    assert ProviderError(K.TIMED_OUT, source="X").retry_after is None


# The HTTP reason phrase of each status, written out (422's name differs between Python versions).
TITLES = {
    422: {"Unprocessable Entity", "Unprocessable Content"},
    502: {"Bad Gateway"},
    503: {"Service Unavailable"},
    504: {"Gateway Timeout"},
}


@pytest.mark.parametrize("kind", sorted(ANSWERS), ids=str)
async def test_the_problem_body_for_each_kind(
    app: FastAPI, client: httpx.AsyncClient, kind: FailureKind
) -> None:
    @app.get("/model")
    async def model() -> None:
        raise ProviderError(kind, source="SECRET") from ValueError("prompt was SECRET")

    status, detail, retry = ANSWERS[kind]
    response = await client.get("/model")
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": response.json()["title"],
        "status": status,
        "detail": detail,
        "instance": "/model",
    }
    assert response.json()["title"] in TITLES[status]
    assert response.headers.get("Retry-After") == ("30" if retry else None)
    assert "SECRET" not in response.text


async def test_retry_after_is_the_providers_when_given_and_only_when_the_outcome_retries(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    errors = {
        "/given": ProviderError(K.RATE_LIMITED, source="x", retry_after="7"),
        "/empty": ProviderError(K.UNREACHABLE, source="x", retry_after=""),
        "/none": ProviderError(K.PROVIDER_FAILED, source="x"),
        "/no-retry": ProviderError(K.TIMED_OUT, source="x", retry_after="9"),
        "/auth": ProviderError(K.AUTHENTICATION, source="x", retry_after="9"),
    }

    def add_route(path: str, error: ProviderError) -> None:
        @app.get(path)
        async def route() -> None:
            raise error

    for path, error in errors.items():
        add_route(path, error)
    sent = {path: (await client.get(path)).headers.get("Retry-After") for path in errors}
    assert sent == {"/given": "7", "/empty": "30", "/none": "30", "/no-retry": None, "/auth": None}


async def test_an_invalid_request_is_a_plain_500_logged_as_an_error(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    error = ProviderError(K.INVALID_REQUEST, source="BadRequestError")

    @app.get("/bug")
    async def bug() -> None:
        raise error

    with capture_logs() as logs:
        response = await client.get("/bug")
    assert response.status_code == 500
    assert response.json() == {
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": 500,
        "instance": "/bug",
    }
    assert [e for e in logs if e["event"] == "llm_error"] == [
        {
            "event": "llm_error",
            "log_level": "error",
            "kind": "BadRequestError",
            "path": "/bug",
            "chain": ["ProviderError"],
        }
    ]


async def test_a_bug_logs_the_type_names_of_the_chain_never_the_text_in_it(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    # A framework's wrapper embeds the provider's text, which can echo the prompt, in the messages
    # of the chain a traceback prints: only the types are logged.
    @app.get("/bug2")
    async def bug() -> None:
        try:
            raise RuntimeError("the prompt said SECRET")
        except RuntimeError as inner:
            raise ProviderError(K.INVALID_REQUEST, source="BadRequestError") from inner

    with capture_logs() as logs:
        await client.get("/bug2")
    [entry] = [e for e in logs if e["event"] == "llm_error"]
    assert entry["chain"] == ["ProviderError", "RuntimeError"]
    assert "SECRET" not in repr(logs)


async def test_a_mapped_provider_error_logs_its_source_as_the_kind(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    # The same event and fields LangChain's errors log (test_llm_error_logs.py).
    @app.get("/busy")
    async def busy() -> None:
        raise ProviderError(K.RATE_LIMITED, source="RateLimitError")

    with capture_logs() as logs:
        await client.get("/busy")
    assert [e for e in logs if e["event"] == "llm_error"] == [
        {
            "event": "llm_error",
            "log_level": "warning",
            "kind": "RateLimitError",
            "status": 503,
            "path": "/busy",
        }
    ]


def test_the_step_limit_has_no_error_or_response_any_more() -> None:
    # An agent now answers at its limit (app/llm/core/steps.py); nothing raises or answers a 422.
    import app.llm.core.errors as errors

    assert not hasattr(errors, "StepLimitReached")
    assert not hasattr(errors, "step_limit_response")


def test_importing_the_core_errors_loads_no_framework() -> None:
    # A fresh interpreter: the test process has LangChain loaded already.
    code = (
        "import sys, app.llm.core.errors\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
