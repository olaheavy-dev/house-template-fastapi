"""OpenAI's embeddings endpoint: the one module of the core that imports a provider SDK (#33).

LangChain's embedding classes discard the provider's usage, which the budget settles from, so the
gateway calls OpenAI's SDK itself, here and only here (llm-contracts.toml lets this module, and no
other, import `openai`). It sends one request and does no budgeting (core/embeddings.py does). An
OpenAI error becomes the core's ProviderError carrying the error's type name, never its message: a
provider's message can echo the input.
"""

from __future__ import annotations

from typing import Any

import openai
from pydantic import SecretStr

from app.llm.core.errors import FailureKind, ProviderError
from app.llm.core.usage import Usage

__all__ = ["KINDS", "create", "kind_of", "openai_client"]

# OpenAI's errors as the core's kinds. Most specific first: APITimeoutError is an
# APIConnectionError, and every one of them an APIError.
KINDS: tuple[tuple[type[Exception], FailureKind], ...] = (
    (openai.AuthenticationError, FailureKind.AUTHENTICATION),
    (openai.PermissionDeniedError, FailureKind.PERMISSION_DENIED),
    (openai.NotFoundError, FailureKind.MODEL_NOT_FOUND),
    (openai.RateLimitError, FailureKind.RATE_LIMITED),
    (openai.BadRequestError, FailureKind.INVALID_REQUEST),
    (openai.APITimeoutError, FailureKind.TIMED_OUT),
    (openai.APIConnectionError, FailureKind.UNREACHABLE),
    (openai.APIError, FailureKind.PROVIDER_FAILED),
)


def openai_client(
    api_key: SecretStr, *, timeout: float, max_retries: int, http_client: Any = None
) -> openai.AsyncOpenAI:
    """OpenAI's client with the gateway's timeout and its SDK retries as the only retry layer.
    `http_client` is a test seam (an httpx client over a fake transport, so the real SDK's retries
    and timeouts run with no network); production passes none."""
    return openai.AsyncOpenAI(
        api_key=api_key.get_secret_value(),
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )


def kind_of(error: BaseException) -> FailureKind | None:
    return next((kind for cls, kind in KINDS if isinstance(error, cls)), None)


async def create(client: Any, model_id: str, texts: list[str]) -> tuple[list[list[float]], Usage]:
    """Send one request. Raises ProviderError from an OpenAI error; any other error passes through
    as itself (it may have been billed: the caller decides)."""
    try:
        response = await client.embeddings.create(model=model_id, input=texts)
    except Exception as error:
        kind = kind_of(error)
        if kind is None:
            raise
        # The type only: a provider's message can echo the input. retry_after is left unset, so the
        # answer carries the default Retry-After, as it always has for embeddings.
        raise ProviderError(kind, source=type(error).__name__) from error
    usage = Usage(input_tokens=int(response.usage.prompt_tokens), output_tokens=0)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)], usage
