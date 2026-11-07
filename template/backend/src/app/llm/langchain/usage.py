"""LangChain's usage_metadata as the core's Usage, read the way the budget has always read it."""

from __future__ import annotations

from typing import Any

from langchain_core.messages.ai import UsageMetadata

from app.llm.core.usage import Usage


def from_langchain(usage: UsageMetadata) -> Usage:
    """The neutral Usage of one call. Both providers report cached tokens inside input_tokens and
    split them out in input_token_details; Anthropic splits cache writes by lifetime
    (langchain-anthropic 1.7.6, which then sets `cache_creation` to 0), OpenAI reports one
    `cache_creation` count with no lifetime (langchain-openai 1.7.2). A detail that is missing,
    None or empty counts as 0, and a count sent as a string ("40") is read as a number. Callers
    handle a call without usage themselves (None, or LangChain's empty dict): this needs one.
    """
    # Untyped on purpose: providers add keys InputTokenDetails doesn't declare (cache lifetimes).
    given: dict[str, Any] = dict(usage.get("input_token_details") or {})
    out: dict[str, Any] = dict(usage.get("output_token_details") or {})

    def count(details: dict[str, Any], key: str) -> int:
        return int(details.get(key) or 0)

    return Usage(
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        reasoning_tokens=count(out, "reasoning"),
        cache_read_tokens=count(given, "cache_read"),
        cache_write_short_tokens=count(given, "ephemeral_5m_input_tokens"),
        cache_write_long_tokens=count(given, "ephemeral_1h_input_tokens"),
        cache_write_undivided_tokens=count(given, "cache_creation"),
    )
