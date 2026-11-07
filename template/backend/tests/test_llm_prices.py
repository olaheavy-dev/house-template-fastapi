from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from app.llm.core.prices import Price, Rates, ceiling, cost
from app.llm.core.usage import Usage
from app.llm.langchain.usage import from_langchain
from app.llm.prices import EMBEDDING_PRICES, PRICES


def usage(input_tokens: int, output_tokens: int, **details: Any) -> Usage:
    return from_langchain(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": details,  # type: ignore[typeddict-item]
        }
    )


CHECKED = "2026-11-06"
SOURCE = "https://example.com/pricing"
SONNET = PRICES["anthropic:claude-sonnet-5-5"]  # $2 in, $10 out, $0.10 read, $2.50 / $4 write
HAIKU = PRICES["anthropic:claude-haiku-5-5"]


def test_prices_are_whole_micro_usd_per_million_tokens() -> None:
    assert SONNET.rates == Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000)


def test_cost_charges_input_and_output_at_their_rates() -> None:
    # 1,000 in at $2/M = 2,000 micro-USD; 500 out at $10/M = 5,000.
    assert cost(SONNET, usage(1_000, 500)) == 7_000


def test_cached_tokens_are_part_of_input_and_charged_at_cache_rates() -> None:
    # LangChain's input_tokens includes cache reads and writes (both providers).
    u = usage(
        1_000, 0, cache_read=600, ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=100
    )
    # 200 plain x 2 + 600 read x 0.10 + 100 x 2.50 + 100 x 4 = 400 + 60 + 250 + 400
    assert cost(SONNET, u) == 1_110


def test_an_undivided_cache_write_is_charged_at_the_longer_lifetime_rate() -> None:
    # langchain-openai 1.7.2 reports one `cache_creation` count with no lifetime split, and
    # OpenAI's pricing page (read 2026-11-06) has a 30m write and a 12h write price. With no way to
    # tell which was used, the dearer one is charged, so cost can only be over-recorded.
    # Sonnet's long write is $4/M: 100 tokens x 4 = 400 micro-USD (not 250 at the short $2.50).
    assert cost(SONNET, usage(100, 0, cache_creation=100)) == 400
    # gpt-6.1-sol, short context: input $2.00, 12h write $5.00. 1,000 input tokens of which 400
    # are cache writes: 600 plain x 2 + 400 x 5 = 1,200 + 2,000 = 3,200 micro-USD.
    sol = PRICES["openai:gpt-6.1-sol"]
    assert cost(sol, usage(1_000, 0, cache_creation=400)) == 3_200


def test_a_split_cache_write_keeps_its_own_lifetime_rates() -> None:
    # Anthropic's integration (langchain-anthropic 1.7.6, _create_usage_metadata) sets
    # `cache_creation` to 0 whenever the 5m/1h split is present, so nothing is counted twice:
    # 100 x 2.50 (5m) + 50 x 4 (1h) = 250 + 200 = 450.
    u = usage(150, 0, cache_creation=0, ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=50)
    assert cost(SONNET, u) == 450


def test_unknown_details_are_charged_as_plain_input() -> None:
    assert cost(SONNET, usage(1_000, 0, priority_cache_read=1_000)) == 2_000


def test_cost_rounds_up_to_a_whole_micro_usd() -> None:
    assert cost(HAIKU, usage(1, 0)) == 1  # 0.1 micro-USD


def test_long_prompts_use_the_long_prompt_rates() -> None:
    assert cost(HAIKU, usage(100_000, 0)) == 10_000
    assert cost(HAIKU, usage(100_001, 0)) == 50_001


def test_ceiling_is_never_below_the_cost_of_the_same_tokens() -> None:
    # Each usage is compared with the ceiling for ITS OWN input and output tokens, so a cost that
    # counted a cache write twice would show. By hand, for the cache_creation-only usage on
    # claude-opus-5-5 ($4/M input, $8/M long write, one tier): correct cost = 1,000 x 8 = 8,000
    # micro-USD, all tokens charged once at the write rate; double counted (also as plain input) =
    # 1,000 x 4 + 1,000 x 8 = 12,000. ceiling(1,000 in, 0 out) = 1,000 x max(4, 8) = 8,000, so it
    # holds the correct cost and would fail the double-counted one.
    for price in PRICES.values():
        for u in (
            usage(1_000, 1_000),
            usage(1_000, 0, ephemeral_1h_input_tokens=1_000),
            usage(1_000, 0, cache_creation=1_000),
        ):
            assert ceiling(price, u.input_tokens, u.output_tokens) >= cost(price, u)


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
    assert cost(HAIKU, usage(0, 0)) == 0


def test_one_token_costs_a_whole_micro_usd_not_a_fraction() -> None:
    # 1 input token x 0.1 = 0.1 -> rounds up to 1. 1 output token x 0.5 = 0.5 -> 1.
    assert cost(HAIKU, usage(1, 0)) == 1
    assert cost(HAIKU, usage(0, 1)) == 1


def test_an_exact_micro_usd_is_not_rounded_up_again() -> None:
    # 10 input tokens x 0.1 = 1.0 exactly -> 1, not 2. 11 x 0.1 = 1.1 -> 2.
    assert cost(HAIKU, usage(10, 0)) == 1
    assert cost(HAIKU, usage(11, 0)) == 2


def test_the_long_prompt_threshold_is_exclusive() -> None:
    # Exactly 100,000 input tokens is still the short tier: 100,000 x 0.1 = 10,000.
    assert cost(HAIKU, usage(100_000, 0)) == 10_000
    # One more switches the WHOLE prompt to the long tier: 100,001 x 0.5 = 50,000.5 -> 50,001.
    assert cost(HAIKU, usage(100_001, 0)) == 50_001


def test_the_long_prompt_tier_prices_output_too() -> None:
    # Short tier, 100,000 in + 3 out: 10,000 + 3 x 0.5 = 10,001.5 -> 10,002.
    assert cost(HAIKU, usage(100_000, 3)) == 10_002
    # Long tier, 100,001 in + 3 out: 50,000.5 + 3 x 2.5 = 50,008 exactly -> 50,008.
    assert cost(HAIKU, usage(100_001, 3)) == 50_008


def latest_plausible_checked(now: datetime) -> date:
    """`checked` is the day the page was read, as the person wrote it: their local date. Someone
    east of UTC (up to UTC+14) is already on the next day for part of the UTC day, so the UTC date
    plus one is the latest a real entry can carry."""
    return now.astimezone(UTC).date() + timedelta(days=1)


def test_a_checked_date_may_be_the_next_utc_day_but_no_later() -> None:
    assert latest_plausible_checked(datetime(2026, 11, 6, 23, 30, tzinfo=UTC)) == date(2026, 11, 7)
    assert latest_plausible_checked(datetime(2026, 11, 6, 0, 0, tzinfo=UTC)) == date(2026, 11, 7)
    # 09:00 on 7 Nov at UTC+14 is 19:00 on 6 Nov in UTC, so the limit is 7 Nov, not 8 Nov.
    east = datetime(2026, 11, 7, 9, 0, tzinfo=timezone(timedelta(hours=14)))
    assert latest_plausible_checked(east) == date(2026, 11, 7)


def test_every_price_records_date_and_source() -> None:
    latest = latest_plausible_checked(datetime.now(UTC))
    for table in (PRICES, EMBEDDING_PRICES):
        for model, price in table.items():
            assert date.fromisoformat(price.checked) <= latest, (
                f"{model}: checked {price.checked} is after {latest} (today in UTC plus one day, "
                "for people east of UTC); `checked` is the day you read the page"
            )
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
