"""How MAF's model errors become the core's failure kinds (#35). MAF wraps the provider SDK's
error and keeps it in `__cause__`, `__context__`, an exception group and `args`; the Anthropic
client calls a 429 an invalid request. The kind is the first SDK error found, whatever wraps it."""

from __future__ import annotations

import email.utils
import time
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from agent_framework.exceptions import (
    ChatClientException,
    ChatClientInvalidAuthException,
    ChatClientInvalidRequestException,
)

from app.llm.core.errors import FailureKind
from app.llm.maf.errors import chain, classify, kind_of, retry_after_of

REQUEST = httpx2.Request("POST", "http://x")


def response(status: int, **headers: str) -> httpx2.Response:
    return httpx2.Response(status, request=REQUEST, headers=headers)


def a(cls: Any, status: int, message: str = "m", **headers: str) -> Exception:
    return cls(message, response=response(status, **headers), body=None)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (a(openai.RateLimitError, 429), FailureKind.RATE_LIMITED),
        (a(anthropic.RateLimitError, 429), FailureKind.RATE_LIMITED),
        (openai.APITimeoutError(request=REQUEST), FailureKind.TIMED_OUT),
        (anthropic.APITimeoutError(request=REQUEST), FailureKind.TIMED_OUT),
        (openai.APIConnectionError(message="x", request=REQUEST), FailureKind.UNREACHABLE),
        (anthropic.APIConnectionError(message="x", request=REQUEST), FailureKind.UNREACHABLE),
        (a(openai.InternalServerError, 500), FailureKind.PROVIDER_FAILED),
        (a(anthropic.InternalServerError, 529), FailureKind.PROVIDER_FAILED),
        (a(openai.AuthenticationError, 401), FailureKind.AUTHENTICATION),
        (a(anthropic.AuthenticationError, 401), FailureKind.AUTHENTICATION),
        (a(openai.PermissionDeniedError, 403), FailureKind.PERMISSION_DENIED),
        (a(anthropic.NotFoundError, 404), FailureKind.MODEL_NOT_FOUND),
        (a(openai.BadRequestError, 400), FailureKind.INVALID_REQUEST),
        (a(anthropic.BadRequestError, 400), FailureKind.INVALID_REQUEST),
        (a(anthropic.UnprocessableEntityError, 422), FailureKind.INVALID_REQUEST),
        (
            a(anthropic.BadRequestError, 400, "prompt is too long: 9 tokens > 5"),
            FailureKind.CONTEXT_OVERFLOW,
        ),
        (a(anthropic.RequestTooLargeError, 413), FailureKind.CONTEXT_OVERFLOW),
        (TimeoutError(), FailureKind.TIMED_OUT),
    ],
)
def test_each_sdk_error_has_its_kind(error: Exception, kind: FailureKind) -> None:
    assert kind_of(error) is kind


def test_an_openai_context_overflow_is_read_from_the_error_code() -> None:
    error = openai.BadRequestError(
        "x", response=response(400), body={"code": "context_length_exceeded"}
    )
    assert error.code == "context_length_exceeded"
    assert kind_of(error) is FailureKind.CONTEXT_OVERFLOW


def test_the_error_is_found_through_cause_context_args_and_groups() -> None:
    inner = a(anthropic.RateLimitError, 429)
    via_cause = ChatClientException("wrapped")
    via_cause.__cause__ = inner
    via_context = RuntimeError("later")
    via_context.__context__ = inner
    via_args = ChatClientException("wrapped", inner_exception=inner)  # Exception("wrapped", inner)
    group = ExceptionGroup(
        "run", [ValueError("a"), ChatClientException("b", inner_exception=inner)]
    )
    for error in (via_cause, via_context, via_args, group):
        assert kind_of(error) is FailureKind.RATE_LIMITED, error


def test_an_anthropic_429_that_arrives_as_an_invalid_request_is_still_a_rate_limit() -> None:
    inner = a(anthropic.RateLimitError, 429)
    wrapper = ChatClientInvalidRequestException("Invalid Anthropic request", inner_exception=inner)
    assert kind_of(wrapper) is FailureKind.RATE_LIMITED


def test_a_wrapper_with_no_sdk_error_behind_it_keeps_its_own_class() -> None:
    assert kind_of(ChatClientInvalidAuthException("x")) is FailureKind.AUTHENTICATION
    assert kind_of(ChatClientInvalidRequestException("x")) is FailureKind.INVALID_REQUEST
    assert kind_of(ChatClientException("x")) is None


def test_an_error_of_no_known_kind_is_not_classified() -> None:
    assert kind_of(RuntimeError("boom")) is None
    assert classify(RuntimeError("boom")) is None


def test_the_chain_survives_a_cycle_and_a_deep_nesting() -> None:
    first, second = RuntimeError("1"), RuntimeError("2")
    first.__cause__, second.__cause__ = second, first
    assert len(list(chain(first))) == 2
    deep: BaseException = RuntimeError("0")
    for number in range(500):
        wrapper = RuntimeError(str(number))
        wrapper.__cause__ = deep
        deep = wrapper
    assert len(list(chain(deep))) <= 64


def test_the_provider_error_carries_the_type_name_never_the_providers_words() -> None:
    inner = a(anthropic.RateLimitError, 429, "slow down: the prompt said SECRET")
    wrapped = ChatClientInvalidRequestException(f"Invalid request: {inner}", inner_exception=inner)
    known = classify(wrapped)
    assert known is not None
    assert (known.kind, known.source) == (FailureKind.RATE_LIMITED, "RateLimitError")
    assert "SECRET" not in str(known)
    assert "SECRET" not in known.source


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, "7"),
        ({"retry-after": "7.2"}, "8"),
        ({"retry-after-ms": "1500"}, "2"),
        ({"retry-after-ms": "1", "retry-after": "30"}, "1"),  # the millisecond hint wins
        ({"retry-after-ms": "junk", "retry-after": "9"}, "9"),
        ({"retry-after": "0"}, "1"),  # at least a second
        ({"retry-after": "junk"}, None),
        ({}, None),
    ],
)
def test_the_providers_retry_hint_is_read_in_whole_seconds(
    headers: dict[str, str], expected: str | None
) -> None:
    error = a(anthropic.RateLimitError, 429, **headers)
    assert retry_after_of(error) == expected


def test_a_retry_hint_given_as_a_date_is_read() -> None:
    when = email.utils.formatdate(time.time() + 120, usegmt=True)
    seconds = int(retry_after_of(a(anthropic.RateLimitError, 429, **{"retry-after": when})) or 0)
    assert 100 <= seconds <= 121
