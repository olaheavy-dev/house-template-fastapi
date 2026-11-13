"""What each model costs, in whole micro-USD (1 USD = 1_000_000) per million tokens.

LangChain reports tokens but no cost, so the gateway prices calls itself. Integers keep the budget
arithmetic exact. A model the gateway may call must be listed here (the gateway refuses to start
otherwise), so add a model's prices before setting LLM_MODEL to it. The template sets no model: you
choose LLM_MODEL and EMBEDDING_MODEL, and these entries are examples to copy from.

Every entry records `checked` (the ISO date you read the price) and `source` (the provider's pricing
page, an https URL), and an entry without both is refused. Prices are the providers' standard-tier
list prices: read the page on the day you add or change an entry and never type one from memory.

A chat entry also records the model's context window (`window`, #46): its size and the most it may
write, read from the provider's model page with its own `checked` and `source`. Without it the
gateway can't say how full a conversation is or trim it; the calls still go through.
"""

from __future__ import annotations

from app.llm.core.prices import MICRO_USD, Price, Rates, Window
from app.llm.core.prices import usd_rates as _rates

__all__ = [
    "EMBEDDING_PRICES",
    "MICRO_USD",
    "PRICES",
    "Price",
    "Rates",
    "Window",
]


CLAUDE_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"
CLAUDE_MODELS = "https://platform.claude.com/docs/en/about-claude/models/overview"
OPENAI_MODELS = "https://developers.openai.com/api/docs/models/"

# Anthropic's page gives "1M tokens" and "128K tokens" for every current model, read as
# 1,000,000 and 128,000 (Window's note).
CLAUDE_WINDOW = Window(1_000_000, 128_000, checked="2026-11-13", source=CLAUDE_MODELS)

# Keyed by the LLM_MODEL value, "provider:model id". USD per million tokens:
# input, output, cache read, cache write (short-lived), cache write (long-lived).
PRICES: dict[str, Price] = {
    "anthropic:claude-opus-5-5": Price(
        _rates(4, 20, 0.20, 5, 8),
        checked="2026-11-06",
        source=CLAUDE_PRICING,
        window=CLAUDE_WINDOW,
    ),
    "anthropic:claude-sonnet-5-5": Price(
        _rates(2, 10, 0.10, 2.50, 4),
        checked="2026-11-06",
        source=CLAUDE_PRICING,
        window=CLAUDE_WINDOW,
    ),
    "anthropic:claude-haiku-5-5": Price(
        _rates(0.10, 0.50, 0.01, 0.125, 0.20),
        checked="2026-11-06",
        source=CLAUDE_PRICING,
        long_prompt_tokens=100_000,
        long_rates=_rates(0.50, 2.50, 0.05, 0.625, 1),
        window=CLAUDE_WINDOW,
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
        window=Window(
            1_050_000, 128_000, checked="2026-11-13", source=OPENAI_MODELS + "gpt-6.1-sol"
        ),
    ),
    "openai:gpt-6-luna": Price(
        _rates(0.10, 0.50, 0.01, 0.125, 0.25),
        checked="2026-11-06",
        source=OPENAI_PRICING,
        max_prompt_tokens=272_000,
        window=Window(
            1_050_000, 128_000, checked="2026-11-13", source=OPENAI_MODELS + "gpt-6-luna"
        ),
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
