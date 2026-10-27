"""What each model costs, in whole micro-USD (1 USD = 1_000_000) per million tokens.

LangChain reports tokens but no cost, so the gateway prices calls itself. Integers keep the budget
arithmetic exact. A model the gateway may call must be listed here (the gateway refuses to start
otherwise), so add a model's prices before switching LLM_MODEL to it.

Prices are the providers' standard-tier list prices, checked on 26 Oct 2026:
https://platform.claude.com/docs/en/about-claude/pricing and
https://developers.openai.com/api/docs/pricing. Re-check them when you change models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages.ai import UsageMetadata

MICRO_USD = 1_000_000
_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Rates:
    input: int
    output: int
    cache_read: int
    # Anthropic prices cache writes by lifetime; OpenAI reports one kind, priced as the shorter.
    cache_write_short: int
    cache_write_long: int


@dataclass(frozen=True, slots=True)
class Price:
    rates: Rates
    # Some models charge more for every token of a request whose prompt is longer than this.
    long_prompt_tokens: int | None = None
    long_rates: Rates | None = None
    # Longest prompt these rates cover, when longer ones cost more and aren't priced here: the
    # budget refuses them instead of under-recording their cost.
    max_prompt_tokens: int | None = None

    def for_prompt(self, input_tokens: int) -> Rates:
        if self.long_rates and self.long_prompt_tokens and input_tokens > self.long_prompt_tokens:
            return self.long_rates
        return self.rates

    @property
    def highest(self) -> Rates:
        return self.long_rates or self.rates


def _rates(input: float, output: float, cache_read: float, short: float, long: float) -> Rates:
    def micro(usd: float) -> int:
        return round(usd * MICRO_USD)

    return Rates(micro(input), micro(output), micro(cache_read), micro(short), micro(long))


# Keyed by the LLM_MODEL value, "provider:model id". USD per million tokens:
# input, output, cache read, cache write (short-lived), cache write (long-lived).
PRICES: dict[str, Price] = {
    "anthropic:claude-opus-5-5": Price(_rates(4, 20, 0.20, 5, 8)),
    "anthropic:claude-sonnet-5-5": Price(_rates(2, 10, 0.10, 2.50, 4)),
    "anthropic:claude-haiku-5-5": Price(
        _rates(0.10, 0.50, 0.01, 0.125, 0.20),
        long_prompt_tokens=100_000,
        long_rates=_rates(0.50, 2.50, 0.05, 0.625, 1),
    ),
    # OpenAI's prices for prompts up to 272K tokens. Longer prompts cost more and aren't priced
    # here, so the budget refuses them (add long_prompt_tokens and long_rates to allow them).
    "openai:gpt-6.1-sol": Price(_rates(2, 10, 0.10, 2.50, 5), max_prompt_tokens=272_000),
    "openai:gpt-6-luna": Price(_rates(0.10, 0.50, 0.01, 0.125, 0.25), max_prompt_tokens=272_000),
}


# Keyed by the EMBEDDING_MODEL value. Embeddings are billed for input tokens only.
# https://developers.openai.com/api/docs/pricing, checked on 27 Oct 2026.
EMBEDDING_PRICES: dict[str, Price] = {
    "openai:text-embedding-3-small": Price(_rates(0.02, 0, 0, 0, 0)),
    "openai:text-embedding-3-large": Price(_rates(0.13, 0, 0, 0, 0)),
}


def cost(price: Price, usage: UsageMetadata) -> int:
    """The cost of one call in micro-USD, rounded up, from the usage LangChain reports.

    input_tokens includes cached tokens for both providers; the details split them out. Tokens in
    a detail this table doesn't know (an OpenAI service tier, audio) are charged as plain input or
    output, which never undercharges a standard-tier call.
    """
    input_tokens = usage["input_tokens"]
    rates = price.for_prompt(input_tokens)
    # Untyped on purpose: providers add keys InputTokenDetails doesn't declare (cache lifetimes).
    details: dict[str, Any] = dict(usage.get("input_token_details") or {})

    def tokens(key: str) -> int:
        return int(details.get(key) or 0)

    cache_read = tokens("cache_read")
    long_write = tokens("ephemeral_1h_input_tokens")
    short_write = tokens("ephemeral_5m_input_tokens") + tokens("cache_creation")
    plain = max(input_tokens - cache_read - long_write - short_write, 0)
    total = (
        plain * rates.input
        + cache_read * rates.cache_read
        + short_write * rates.cache_write_short
        + long_write * rates.cache_write_long
        + usage["output_tokens"] * rates.output
    )
    return -(-total // _PER_MILLION)


def ceiling(price: Price, input_tokens: int, output_tokens: int) -> int:
    """The most a call can cost: every input token uncached at the highest rate, every output token
    the call may generate. The budget reserves this before the call."""
    rates = price.highest
    total = input_tokens * max(rates.input, rates.cache_write_long) + output_tokens * rates.output
    return -(-total // _PER_MILLION)


def token_details(usage: UsageMetadata | None) -> dict[str, int | None]:
    """Reasoning, cache-read and cache-write tokens from LangChain's usage details (both providers;
    Anthropic splits cache writes by lifetime)."""
    if not usage:
        return {"reasoning_tokens": None, "cache_read_tokens": None, "cache_write_tokens": None}
    given: dict[str, Any] = dict(usage.get("input_token_details") or {})
    out: dict[str, Any] = dict(usage.get("output_token_details") or {})
    writes = sum(
        int(given.get(key) or 0)
        for key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    return {
        "reasoning_tokens": int(out.get("reasoning") or 0),
        "cache_read_tokens": int(given.get("cache_read") or 0),
        "cache_write_tokens": writes,
    }
