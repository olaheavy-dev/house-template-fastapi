"""MAF's model errors as the core's failure kinds (app/llm/core/errors.py).

MAF wraps the provider SDK's error and keeps it in three places: `__cause__` (`raise ... from`),
`__context__`, and the exception's own `args` (`AgentFrameworkException(message, inner_exception)`
is `Exception(message, inner)`), and an agent run may group several in an ExceptionGroup. Its
Anthropic client never retries and maps a 429 to an invalid-request error, and its OpenAI client
wraps everything in one ChatClientException, so the type of the error MAF raises says nothing
useful: `chain` walks all four places and `kind_of` classifies the first SDK error it finds.

The answer to the client is the core's, decided by the kind (core/errors.py OUTCOMES) and never
the provider's text: MAF's own messages embed the provider's (`f"... failed: {ex}"`), which can
echo the prompt, so nothing from `str(error)` or `args` text is read here, logged or returned.
`classify` makes a ProviderError, raised `from` the original; the SDK's retries are the only retry
layer (the gateway builds the client with `max_retries`), so an error that arrives here has
already been retried.
"""

from __future__ import annotations

import email.utils
import math
import time
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

import anthropic
import openai
from agent_framework.exceptions import (
    ChatClientInvalidAuthException,
    ChatClientInvalidRequestException,
)

from app.llm.core.errors import FailureKind, ProviderError

__all__ = ["chain", "classify", "kind_of", "retry_after_of"]

# Most errors one chain may visit: a self-referencing group cannot loop (ids are tracked), and a
# pathological one cannot make a failed call slow.
_MAX_ERRORS = 64

# Context overflow is a 400 whose body says so; the code and phrase are the providers' documented
# ones (OpenAI: error code `context_length_exceeded`; Anthropic: "prompt is too long: N tokens >
# M maximum", and a 413 `request_too_large`). They are read, never repeated.
_OPENAI_OVERFLOW = "context_length_exceeded"
_ANTHROPIC_OVERFLOW = "prompt is too long"


def chain(error: BaseException | None) -> Iterator[BaseException]:
    """The error and everything behind it, outermost first: `__cause__`, `__context__`, the
    members of an exception group, and any exception held in `args`."""
    seen: set[int] = set()
    pending: deque[BaseException] = deque([error] if error is not None else [])
    while pending and len(seen) < _MAX_ERRORS:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for behind in (current.__cause__, current.__context__, *current.args):
            if isinstance(behind, BaseException):
                pending.append(behind)


def _status_of(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _overflow(error: BaseException) -> bool:
    """A 400 or 413 that says the input is longer than the model's context."""
    if isinstance(error, anthropic.RequestTooLargeError):
        return True
    if isinstance(error, openai.BadRequestError):
        return getattr(error, "code", None) == _OPENAI_OVERFLOW
    if isinstance(error, anthropic.BadRequestError):
        return _ANTHROPIC_OVERFLOW in str(error.message).lower()
    return False


def _kind_of_one(error: BaseException) -> FailureKind | None:
    """The kind of one error, None when it is not a provider SDK's (or MAF's own typed) error.
    Most specific first: a timeout is a connection error, and every status error is an API error
    in both SDKs."""
    if _overflow(error):
        return FailureKind.CONTEXT_OVERFLOW
    if isinstance(error, openai.AuthenticationError | anthropic.AuthenticationError):
        return FailureKind.AUTHENTICATION
    if isinstance(error, openai.PermissionDeniedError | anthropic.PermissionDeniedError):
        return FailureKind.PERMISSION_DENIED
    if isinstance(error, openai.NotFoundError | anthropic.NotFoundError):
        return FailureKind.MODEL_NOT_FOUND
    if isinstance(error, openai.RateLimitError | anthropic.RateLimitError):
        return FailureKind.RATE_LIMITED
    if isinstance(error, openai.BadRequestError | anthropic.BadRequestError):
        return FailureKind.INVALID_REQUEST
    if isinstance(error, openai.APITimeoutError | anthropic.APITimeoutError | TimeoutError):
        return FailureKind.TIMED_OUT
    if isinstance(error, openai.APIConnectionError | anthropic.APIConnectionError):
        return FailureKind.UNREACHABLE
    if isinstance(error, openai.APIError | anthropic.APIError):
        # A status error of another 4xx (409, 422) is the app's own request again.
        status = _status_of(error)
        if status is not None and 400 <= status < 500:
            return FailureKind.INVALID_REQUEST
        return FailureKind.PROVIDER_FAILED
    return None


def _found(error: BaseException) -> tuple[FailureKind, BaseException] | None:
    """The first error in the chain with a kind, and its kind. A provider SDK's error beats MAF's
    own typed wrappers, which carry no more than a status class (and the Anthropic client's
    wrapper calls a 429 an invalid request): the wrapper counts only when no SDK error is behind
    it."""
    wrappers: list[tuple[FailureKind, BaseException]] = []
    for link in chain(error):
        kind = _kind_of_one(link)
        if kind is not None:
            return kind, link
        if isinstance(link, ChatClientInvalidAuthException):
            wrappers.append((FailureKind.AUTHENTICATION, link))
        elif isinstance(link, ChatClientInvalidRequestException):
            wrappers.append((FailureKind.INVALID_REQUEST, link))
    return wrappers[0] if wrappers else None


def kind_of(error: BaseException) -> FailureKind | None:
    """The kind of a failed call's error, found anywhere in its chain; None for anything else (a
    bug, a library error, a budget refusal): the budget keeps the reservation and the client gets
    a plain 500."""
    found = _found(error)
    return found[0] if found else None


def _seconds(headers: Mapping[str, Any]) -> float | None:
    """The provider's retry hint in seconds: `retry-after-ms` first (both SDKs read it), then
    `retry-after` as seconds or as an HTTP date."""
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        try:
            return float(milliseconds) / 1000
        except (TypeError, ValueError):
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        parsed = email.utils.parsedate_tz(str(value))
        return None if parsed is None else email.utils.mktime_tz(parsed) - time.time()


def retry_after_of(error: BaseException) -> str | None:
    """The provider's Retry-After as whole seconds (at least 1), read from the response of the
    first error in the chain that has one; None when the provider gave none."""
    for link in chain(error):
        headers = getattr(getattr(link, "response", None), "headers", None)
        if headers is None:
            continue
        seconds = _seconds(headers)
        if seconds is not None and math.isfinite(seconds):
            return str(max(1, math.ceil(seconds)))
    return None


def classify(error: BaseException) -> ProviderError | None:
    """The core's ProviderError for a MAF call's error, to be raised `from` it; None when the
    error is of no known kind (it passes through as itself)."""
    found = _found(error)
    if found is None:
        return None
    kind, link = found
    return ProviderError(kind, source=type(link).__name__, retry_after=retry_after_of(error))
