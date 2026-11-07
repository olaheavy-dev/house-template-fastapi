"""LangChain's model errors as the core's failure kinds (#33 T2): the classification order, what
the budget releases, the provider's Retry-After, and the handlers the app installs."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_anthropic.chat_models import (
    AnthropicContextOverflowError,
    AnthropicInvalidRequestError,
    AnthropicRateLimitError,
)
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
from langchain_openai.chat_models.base import (
    OpenAIAPIContextOverflowError,
    OpenAIAPIError,
    OpenAIAuthenticationError,
    OpenAIConnectionError,
    OpenAIContextOverflowError,
    OpenAIInvalidRequestError,
    OpenAIModelNotFoundError,
    OpenAIPermissionDeniedError,
    OpenAIRateLimitError,
    OpenAITimeoutError,
)
from langgraph.errors import GraphRecursionError

from app.llm.core.errors import REJECTED, FailureKind, ProviderError, StepLimitReached
from app.llm.langchain.errors import KINDS, kind_of, model_error, retry_after_of

K = FailureKind


def test_the_classification_table_in_this_order() -> None:
    assert (
        (ModelRateLimitError, K.RATE_LIMITED),
        (ModelTimeoutError, K.TIMED_OUT),
        (ContextOverflowError, K.CONTEXT_OVERFLOW),
        (ModelConnectionError, K.UNREACHABLE),
        (ModelAPIError, K.PROVIDER_FAILED),
        (ModelAuthenticationError, K.AUTHENTICATION),
        (ModelPermissionDeniedError, K.PERMISSION_DENIED),
        (ModelNotFoundError, K.MODEL_NOT_FOUND),
        (ModelInvalidRequestError, K.INVALID_REQUEST),
    ) == KINDS


def blank(cls: type[BaseException]) -> BaseException:
    """An instance without running the provider SDK's constructor (it wants a real response)."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (OpenAIRateLimitError, K.RATE_LIMITED),
        (OpenAITimeoutError, K.TIMED_OUT),
        (OpenAIContextOverflowError, K.CONTEXT_OVERFLOW),
        # An openai.APIError, not a ModelAPIError: still a context overflow.
        (OpenAIAPIContextOverflowError, K.CONTEXT_OVERFLOW),
        (OpenAIConnectionError, K.UNREACHABLE),
        (OpenAIAPIError, K.PROVIDER_FAILED),
        (OpenAIAuthenticationError, K.AUTHENTICATION),
        (OpenAIPermissionDeniedError, K.PERMISSION_DENIED),
        (OpenAIModelNotFoundError, K.MODEL_NOT_FOUND),
        (OpenAIInvalidRequestError, K.INVALID_REQUEST),
        (AnthropicRateLimitError, K.RATE_LIMITED),
        (AnthropicContextOverflowError, K.CONTEXT_OVERFLOW),
        (AnthropicInvalidRequestError, K.INVALID_REQUEST),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_the_provider_integrations_classes_classify_by_their_family(
    cls: type[BaseException], kind: FailureKind
) -> None:
    assert kind_of(blank(cls)) is kind


def test_the_first_row_wins_over_the_class_order() -> None:
    class Both(ModelInvalidRequestError, ModelRateLimitError):
        pass

    class Overflow(ModelInvalidRequestError, ContextOverflowError):
        pass

    # The table's order decides, not the bases': rate limit and overflow come before invalid.
    assert kind_of(Both("x")) is K.RATE_LIMITED
    assert kind_of(Overflow("x")) is K.CONTEXT_OVERFLOW


class ApiAndInvalid(ModelAPIError, ModelInvalidRequestError):
    """A class in two families, one rejected (invalid request) and one that may have been billed."""


class RateLimitAndApi(ModelRateLimitError, ModelAPIError):
    pass


def test_a_class_in_two_families_is_decided_by_the_first_matching_row() -> None:
    # Main released a reservation if ANY base was a rejected class. Now the first matching row of
    # KINDS decides: ModelAPIError comes before ModelInvalidRequestError, so this keeps its
    # reservation. That is the safer side: it can over-record spend, never under-record it.
    assert kind_of(ApiAndInvalid("x")) is K.PROVIDER_FAILED
    assert kind_of(ApiAndInvalid("x")) not in REJECTED
    # Rate limit comes first, so the refund follows it, as it did on main.
    assert kind_of(RateLimitAndApi("x")) is K.RATE_LIMITED
    assert kind_of(RateLimitAndApi("x")) in REJECTED


@pytest.mark.parametrize(
    "error",
    [
        ModelError("plain"),
        KeyError("x"),
        GraphRecursionError("x"),
        ProviderError(K.RATE_LIMITED, source="x"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_anything_else_has_no_kind(error: BaseException) -> None:
    assert kind_of(error) is None


def test_the_budget_releases_exactly_the_classes_it_always_did() -> None:
    # budget.py's _REJECTED before #33 T2, written out.
    released = {
        ModelAuthenticationError,
        ModelPermissionDeniedError,
        ModelInvalidRequestError,
        ModelNotFoundError,
        ModelRateLimitError,
        ContextOverflowError,
    }
    every = {cls for cls, _ in KINDS} | {ModelError}
    assert {cls for cls in every if kind_of(cls("x")) in REJECTED} == released


@pytest.mark.parametrize("kind", list(FailureKind), ids=str)
def test_model_error_gives_the_class_of_each_kind(kind: FailureKind) -> None:
    error = model_error(kind, "OpenAI embeddings: RateLimitError")
    assert kind_of(error) is kind
    assert type(error) in {cls for cls, _ in KINDS}
    assert str(error) == "OpenAI embeddings: RateLimitError"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (SimpleNamespace(headers={"retry-after": "7"}), "7"),
        (SimpleNamespace(headers=httpx.Headers({"Retry-After": "8"})), "8"),
        (SimpleNamespace(headers={"retry-after": 12}), "12"),
        (SimpleNamespace(headers={"retry-after": ""}), None),
        (SimpleNamespace(headers={}), None),
        (SimpleNamespace(headers=None), None),
        (None, None),
    ],
)
def test_retry_after_is_read_from_the_providers_response(
    response: object, expected: str | None
) -> None:
    error = ModelRateLimitError("x")
    if response is not None:
        error.response = response  # type: ignore[attr-defined]
    assert retry_after_of(error) == expected


def test_the_app_answers_the_neutral_and_langchains_failures(app: FastAPI) -> None:
    assert {ProviderError, StepLimitReached, ModelError, GraphRecursionError} <= set(
        app.exception_handlers
    )
