"""What each model costs, in whole micro-USD (1 USD = 1_000_000) per million tokens.

LangChain reports tokens but no cost, so the gateway prices calls itself. Integers keep the budget
arithmetic exact. A model the gateway may call must be listed here (the gateway refuses to start
otherwise), so add a model's prices before setting LLM_MODEL to it. The template sets no model: you
choose LLM_MODEL and EMBEDDING_MODEL, and these entries are examples to copy from.

Every entry records `checked` (the ISO date you read the price) and `source` (the provider's pricing
page, an https URL), and an entry without both is refused. Prices are the providers' standard-tier
list prices: read the page on the day you add or change an entry and never type one from memory.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import prices as _core
from app.llm.core.prices import MICRO_USD, Price, Rates
from app.llm.core.prices import usd_rates as _rates

if TYPE_CHECKING:
    from app.llm.core.prices import ceiling as ceiling
    from app.llm.core.prices import cost as cost
    from app.llm.core.prices import token_details as token_details

__all__ = [
    "EMBEDDING_PRICES",
    "MICRO_USD",
    "PRICES",
    "Price",
    "Rates",
    "ceiling",
    "cost",
    "token_details",
]


CLAUDE_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"

# Keyed by the LLM_MODEL value, "provider:model id". USD per million tokens:
# input, output, cache read, cache write (short-lived), cache write (long-lived).
PRICES: dict[str, Price] = {
    "anthropic:claude-opus-5-5": Price(
        _rates(4, 20, 0.20, 5, 8), checked="2026-11-06", source=CLAUDE_PRICING
    ),
    "anthropic:claude-sonnet-5-5": Price(
        _rates(2, 10, 0.10, 2.50, 4), checked="2026-11-06", source=CLAUDE_PRICING
    ),
    "anthropic:claude-haiku-5-5": Price(
        _rates(0.10, 0.50, 0.01, 0.125, 0.20),
        checked="2026-11-06",
        source=CLAUDE_PRICING,
        long_prompt_tokens=100_000,
        long_rates=_rates(0.50, 2.50, 0.05, 0.625, 1),
    ),
    # OpenAI's prices for prompts up to 272K tokens. Longer prompts cost more and aren't priced
    # here, so the budget refuses them (add long_prompt_tokens and long_rates to allow them). The
    # page defines short context as "<=272K input tokens" without saying what K is; 272_000 reads it
    # as 1,000. The last two rates are the page's 30-minute and 12-hour cache-write columns.
    "openai:gpt-6.1-sol": Price(
        _rates(2, 10, 0.10, 2.50, 5),
        checked="2026-11-06",
        source=OPENAI_PRICING,
        max_prompt_tokens=272_000,
    ),
    "openai:gpt-6-luna": Price(
        _rates(0.10, 0.50, 0.01, 0.125, 0.25),
        checked="2026-11-06",
        source=OPENAI_PRICING,
        max_prompt_tokens=272_000,
    ),
}


# Keyed by the EMBEDDING_MODEL value. Embeddings are billed for input tokens only.
EMBEDDING_PRICES: dict[str, Price] = {
    "openai:text-embedding-3-small": Price(
        _rates(0.02, 0, 0, 0, 0), checked="2026-11-06", source=OPENAI_PRICING
    ),
    "openai:text-embedding-3-large": Price(
        _rates(0.13, 0, 0, 0, 0), checked="2026-11-06", source=OPENAI_PRICING
    ),
}


# The arithmetic moved to app/llm/core/prices.py (#33). Until 0.8.0 the old names still answer here,
# with a DeprecationWarning, and are the same objects as the core's.
_MOVED = frozenset({"cost", "ceiling", "token_details"})

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            warnings.warn(
                f"app.llm.prices.{name} moved to app.llm.core.prices.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(_core, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
