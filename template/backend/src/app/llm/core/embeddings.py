"""Embeddings under the daily model budget, whatever framework asks for them (#33).

A list of texts is split into batches that fit one request, and each batch is one metered call on a
Meter (core/budget.py): reserved before it is sent, sized by its UTF-8 bytes (which bound its
tokens) plus the margin, and settled from the usage the provider returns. A request the provider
rejected gives its reservation back (core/errors.py REJECTED); any other failure keeps it. The
price's max_prompt_tokens applies to each request: texts are batched under it, and one text past it
alone is refused (PromptTooLong, a 422) before anything is reserved or sent.

Personal data is removed first, one way (#78, docs/decisions/0006): with a Redactor, each text is
redacted on its own (Redactor.redact_each), so a text's vector depends only on the text and the
redaction scope, never on its neighbours or on where the batches are cut. Vectors come back, so
nothing is restored and no mapping is kept. The order is: redact every text, count the values
replaced (the llm.redactions metric, once for the call), check the size of the redacted texts, cut
batches on the redacted texts, then for each batch reserve with that batch's own count, send,
settle. A text that cannot be redacted fails the whole call before anything is reserved or sent
(RedactionFailed, a 503). The redactor is required: None means redaction is off (the row's count
stays NULL); forgetting it is a TypeError, not a silent leak.

The provider call is passed in (core/openai_embeddings.py is OpenAI's), so no SDK is imported here.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence

from app.llm.core import metrics
from app.llm.core.bounds import INPUT_MARGIN_TOKENS, check_prompt_size
from app.llm.core.budget import CallTags, Meter
from app.llm.core.errors import ProviderError
from app.llm.core.redaction import Redactor
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
    redactor: Redactor | None,
    tags: CallTags | None = None,
    max_inputs: int = MAX_INPUTS,
    max_bytes: int = MAX_BYTES,
) -> list[list[float]]:
    """The vectors of `texts`, in order, of the texts with personal data removed when there is a
    `redactor` (None: redaction is off). Raises RedactionFailed before anything is sent when a text
    cannot be redacted; PromptTooLong, before anything is sent, for a (redacted) text past the
    price's max_prompt_tokens; BudgetExhausted or UserBudgetExhausted when a batch is refused (the
    earlier ones are already sent and settled); and whatever `create` raises (a ProviderError for a
    provider's error)."""
    counts: list[int] | None = None
    if redactor is not None:
        redacted = redactor.redact_each(texts)  # raises RedactionFailed: the call stops here
        metrics.record_redactions(redacted.kinds)
        texts, counts = redacted.texts, list(redacted.counts)
    limit = meter.price.max_prompt_tokens
    if limit:  # 0 is no limit, as in bounds.check_prompt_size
        max_bytes = min(max_bytes, limit - INPUT_MARGIN_TOKENS)
        # Every text first: one past the limit refuses the call before any earlier batch is sent
        # and paid for.
        for text in texts:
            check_prompt_size(meter.price, len(text.encode()) + INPUT_MARGIN_TOKENS)
    vectors: list[list[float]] = []
    start = 0
    for batch in batches(texts, max_inputs=max_inputs, max_bytes=max_bytes):
        batch_tags = tags
        if counts is not None:
            # This batch's own count, not the call's: each row says what was replaced in it.
            batch_tags = dataclasses.replace(
                tags or CallTags(), redacted_count=sum(counts[start : start + len(batch)])
            )
        start += len(batch)
        vectors += await _embed(meter, create, batch, batch_tags)
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
