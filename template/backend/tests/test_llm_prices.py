from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from langchain_core.messages.ai import UsageMetadata

from app.llm import prices
from app.llm.prices import EMBEDDING_PRICES, PRICES, Price, Rates


def usage(input_tokens: int, output_tokens: int, **details: Any) -> UsageMetadata:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": details,  # type: ignore[typeddict-item]
    }


CHECKED = "2026-11-06"
SOURCE = "https://example.com/pricing"
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
    price = Price(Rates(1, 1, 1, 1, 1), checked=CHECKED, source=SOURCE)
    assert price.for_prompt(10**9) is price.rates is price.highest


# Cost rounds UP to a whole micro-USD. Haiku's rates are worked out by hand from its list prices
# (USD per million tokens, so micro-USD per token): input $0.10 = 0.1, output $0.50 = 0.5 up to
# 100,000 prompt tokens; input $0.50 = 0.5, output $2.50 = 2.5 above that. The cost of a call is
# tokens x rate, rounded up.


def test_zero_tokens_cost_nothing() -> None:
    # 0 x 0.1 + 0 x 0.5 = 0 exactly: rounding up must not turn nothing into 1.
    assert prices.cost(HAIKU, usage(0, 0)) == 0


def test_one_token_costs_a_whole_micro_usd_not_a_fraction() -> None:
    # 1 input token x 0.1 = 0.1 -> rounds up to 1. 1 output token x 0.5 = 0.5 -> 1.
    assert prices.cost(HAIKU, usage(1, 0)) == 1
    assert prices.cost(HAIKU, usage(0, 1)) == 1


def test_an_exact_micro_usd_is_not_rounded_up_again() -> None:
    # 10 input tokens x 0.1 = 1.0 exactly -> 1, not 2. 11 x 0.1 = 1.1 -> 2.
    assert prices.cost(HAIKU, usage(10, 0)) == 1
    assert prices.cost(HAIKU, usage(11, 0)) == 2


def test_the_long_prompt_threshold_is_exclusive() -> None:
    # Exactly 100,000 input tokens is still the short tier: 100,000 x 0.1 = 10,000.
    assert prices.cost(HAIKU, usage(100_000, 0)) == 10_000
    # One more switches the WHOLE prompt to the long tier: 100,001 x 0.5 = 50,000.5 -> 50,001.
    assert prices.cost(HAIKU, usage(100_001, 0)) == 50_001


def test_the_long_prompt_tier_prices_output_too() -> None:
    # Short tier, 100,000 in + 3 out: 10,000 + 3 x 0.5 = 10,001.5 -> 10,002.
    assert prices.cost(HAIKU, usage(100_000, 3)) == 10_002
    # Long tier, 100,001 in + 3 out: 50,000.5 + 3 x 2.5 = 50,008 exactly -> 50,008.
    assert prices.cost(HAIKU, usage(100_001, 3)) == 50_008


def test_every_price_records_date_and_source() -> None:
    for table in (PRICES, EMBEDDING_PRICES):
        for model, price in table.items():
            assert date.fromisoformat(price.checked) <= date.today(), model
            assert price.source.startswith("https://"), model


def test_a_price_without_a_date_or_source_is_refused() -> None:
    rates = Rates(1, 1, 1, 1, 1)
    with pytest.raises(TypeError):
        Price(rates)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Price(rates, checked=CHECKED)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Price(rates, source=SOURCE)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="checked"):
        Price(rates, checked="", source=SOURCE)
    with pytest.raises(ValueError, match="checked"):
        Price(rates, checked="26 Oct 2026", source=SOURCE)
    # Only the exact YYYY-MM-DD form: other spellings Python's fromisoformat accepts are refused.
    for spelling in ("20261106", "2026-W45-5", "2026-11-6", " 2026-11-06"):
        with pytest.raises(ValueError, match="checked"):
            Price(rates, checked=spelling, source=SOURCE)
    with pytest.raises(ValueError, match="in the future"):
        Price(rates, checked=(date.today() + timedelta(days=1)).isoformat(), source=SOURCE)
    with pytest.raises(ValueError, match="source"):
        Price(rates, checked=CHECKED, source="")
    with pytest.raises(ValueError, match="source"):
        Price(rates, checked=CHECKED, source="the pricing page")
    for url in ("https://", "https:///pricing", "http://example.com/pricing", "ftp://example.com"):
        with pytest.raises(ValueError, match="source"):
            Price(rates, checked=CHECKED, source=url)


def test_openai_prompt_limit_is_the_pages_272k_read_as_272_000() -> None:
    # Assumption: OpenAI's page says "Short context: <=272K input tokens" without defining K. This
    # template reads it as 1,000, so 272,000 is the last prompt priced at the short rates. If K is
    # 1,024 the limit would be 278,528 and these entries would be over-cautious, never under.
    for model in ("openai:gpt-6.1-sol", "openai:gpt-6-luna"):
        assert PRICES[model].max_prompt_tokens == 272_000
