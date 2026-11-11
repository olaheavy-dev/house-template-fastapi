"""MAF's usage_details as the core's Usage (agent-framework-core 1.25.0).

UsageDetails keys: input_token_count (the total prompt tokens, cached ones included),
output_token_count (reasoning tokens included), reasoning_output_token_count,
cache_read_input_token_count and cache_creation_input_token_count. The OpenAI Responses client
fills them from the response's usage; the Anthropic client adds cache creation and cache reads to
the input total (so input_token_count is already the total the core wants) and reports a cache
write with no lifetime, so it is charged as an undivided write at the longer lifetime's rate: a
deliberate over-count, never an under-count (app/llm/core/prices.py cost()).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.core.usage import Usage

__all__ = ["from_maf"]


def _count(value: Any) -> int | None:
    """A whole, non-negative number of tokens; None for anything else (a bool is not a count)."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def from_maf(details: Mapping[str, Any] | None) -> Usage | None:
    """The neutral Usage of one call, or None when the provider's usage is missing or cannot be
    trusted: no details, no input or output total, or a count that is not a whole non-negative
    number. None makes the budget keep the call's reservation as its cost (an over-count) rather
    than record a cheap call that may not have been. A detail that is missing or None counts as 0,
    as it always has (Usage)."""
    if not details:
        return None
    given = _count(details.get("input_token_count"))
    produced = _count(details.get("output_token_count"))
    if given is None or produced is None:
        return None
    extras: dict[str, int] = {}
    for key in (
        "reasoning_output_token_count",
        "cache_read_input_token_count",
        "cache_creation_input_token_count",
    ):
        value = details.get(key)
        if value is None:
            extras[key] = 0
            continue
        counted = _count(value)
        if counted is None:
            return None
        extras[key] = counted
    return Usage(
        input_tokens=given,
        output_tokens=produced,
        reasoning_tokens=extras["reasoning_output_token_count"],
        cache_read_tokens=extras["cache_read_input_token_count"],
        cache_write_undivided_tokens=extras["cache_creation_input_token_count"],
    )
