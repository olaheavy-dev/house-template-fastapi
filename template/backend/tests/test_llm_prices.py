from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages.ai import UsageMetadata

from app.llm import prices
from app.llm.prices import PRICES, Price, Rates


def usage(input_tokens: int, output_tokens: int, **details: Any) -> UsageMetadata:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": details,  # type: ignore[typeddict-item]
    }


SONNET = PRICES["anthropic:claude-sonnet-5-5"]  # $2 in, $10 out, $0.10 read, $2.50 / $4 write
HAIKU = PRICES["anthropic:claude-haiku-5-5"]


def test_prices_are_whole_micro_usd_per_million_tokens() -> None:
    assert SONNET.rates == Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000)


def test_cost_charges_input_and_output_at_their_rates() -> None:
    # 1,000 in at $2/M = 2,000 micro-USD; 500 out at $10/M = 5,000.
    assert prices.cost(SONNET, usage(1_000, 500)) == 7_000


def test_cached_tokens_are_part_of_input_and_charged_at_cache_rates() -> None:
    # LangChain's input_tokens includes cache reads and writes (both providers).
    u = usage(
        1_000, 0, cache_read=600, ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=100
    )
    # 200 plain x 2 + 600 read x 0.10 + 100 x 2.50 + 100 x 4 = 400 + 60 + 250 + 400
    assert prices.cost(SONNET, u) == 1_110


def test_openai_cache_writes_are_charged_as_short_lived() -> None:
    assert prices.cost(SONNET, usage(100, 0, cache_creation=100)) == 250


def test_unknown_details_are_charged_as_plain_input() -> None:
    assert prices.cost(SONNET, usage(1_000, 0, priority_cache_read=1_000)) == 2_000


def test_cost_rounds_up_to_a_whole_micro_usd() -> None:
    assert prices.cost(HAIKU, usage(1, 0)) == 1  # 0.1 micro-USD


def test_long_prompts_use_the_long_prompt_rates() -> None:
    assert prices.cost(HAIKU, usage(100_000, 0)) == 10_000
    assert prices.cost(HAIKU, usage(100_001, 0)) == 50_001


def test_ceiling_is_never_below_the_cost_of_the_same_tokens() -> None:
    for price in PRICES.values():
        for u in (usage(1_000, 1_000), usage(1_000, 0, ephemeral_1h_input_tokens=1_000)):
            assert prices.ceiling(price, 1_000, 1_000) >= prices.cost(price, u)


@pytest.mark.parametrize("model", sorted(PRICES))
def test_every_listed_model_is_provider_prefixed(model: str) -> None:
    provider, _, name = model.partition(":")
    assert provider in {"anthropic", "openai"}
    assert name


def test_a_price_without_long_rates_has_one_tier() -> None:
    price = Price(Rates(1, 1, 1, 1, 1))
    assert price.for_prompt(10**9) is price.rates is price.highest
