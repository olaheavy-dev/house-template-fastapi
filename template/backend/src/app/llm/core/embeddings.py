"""Embeddings under the daily model budget, whatever framework asks for them (#33).

A list of texts is split into batches that fit one request, and each batch is one metered call on a
Meter (core/budget.py): reserved before it is sent, sized by its UTF-8 bytes (which bound its
tokens) plus the margin, and settled from the usage the provider returns. A request the provider
rejected gives its reservation back (core/errors.py REJECTED); any other failure keeps it. The
price's max_prompt_tokens applies to each request: texts are batched under it, and one text past it
alone is refused (PromptTooLong, a 422) before anything is reserved or sent.

The provider call is passed in (core/openai_embeddings.py is OpenAI's), so no SDK is imported here.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence

from app.llm.core.bounds import INPUT_MARGIN_TOKENS
from app.llm.core.budget import CallTags, Meter
from app.llm.core.errors import ProviderError
from app.llm.core.usage import Usage

__all__ = ["MAX_BYTES", "MAX_INPUTS", "Create", "batches", "embed"]

# OpenAI's limits per request: 2,048 inputs and 300,000 tokens (bytes bound tokens).
MAX_INPUTS = 2048
MAX_BYTES = 300_000 - INPUT_MARGIN_TOKENS

# Sends one batch: its vectors, in input order, and the usage the provider reported.
Create = Callable[[list[str]], Awaitable[tuple[list[list[float]], Usage]]]


def batches(
    texts: Sequence[str], *, max_inputs: int = MAX_INPUTS, max_bytes: int = MAX_BYTES
) -> Iterator[list[str]]:
    """Consecutive texts in batches of at most max_inputs and max_bytes of UTF-8. A text longer
    than max_bytes is alone in its batch; no batch is ever empty."""
    batch: list[str] = []
    size = 0
    for text in texts:
        length = len(text.encode())
        if batch and (len(batch) == max_inputs or size + length > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += length
    if batch:
        yield batch


async def embed(
    meter: Meter,
    create: Create,
    texts: Sequence[str],
    *,
    tags: CallTags | None = None,
    max_inputs: int = MAX_INPUTS,
    max_bytes: int = MAX_BYTES,
) -> list[list[float]]:
    """The vectors of `texts`, in order. Raises BudgetExhausted or UserBudgetExhausted when a batch
    is refused (the earlier ones are already sent and settled), PromptTooLong for a text past the
    price's max_prompt_tokens, and whatever `create` raises (a ProviderError for a provider's
    error)."""
    limit = meter.price.max_prompt_tokens
    if limit:  # 0 is no limit, as in bounds.check_prompt_size
        max_bytes = min(max_bytes, limit - INPUT_MARGIN_TOKENS)
    vectors: list[list[float]] = []
    for batch in batches(texts, max_inputs=max_inputs, max_bytes=max_bytes):
        vectors += await _embed(meter, create, batch, tags)
    return vectors


async def _embed(
    meter: Meter, create: Create, batch: list[str], tags: CallTags | None
) -> list[list[float]]:
    call_id = uuid.uuid4()
    bound = sum(len(text.encode()) for text in batch) + INPUT_MARGIN_TOKENS
    await meter.reserve(call_id, input_tokens=bound, output_tokens=0, tags=tags)
    try:
        vectors, usage = await create(batch)
    except ProviderError as error:
        await meter.failed(call_id, error.kind, streamed=False)
        raise
    except Exception:
        # Unknown: it may have been billed, so the reservation stays as its cost.
        await meter.failed(call_id, None, streamed=False)
        raise
    except BaseException:
        # Cancelled: the call may still be running at the provider, so its row stays reserved until
        # expire_stale closes it. Nothing is awaited here, which would be cancelled again.
        meter.abandon(call_id)
        raise
    await meter.settle(call_id, usage)
    return vectors
