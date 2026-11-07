"""Characterisation of the cost arithmetic (#33): what prices.cost, ceiling and token_details return
today, so moving them into a framework-neutral core cannot change a figure.

Every expected value is worked out by hand from the list prices in the comments, not by calling
the code under test. Units: a rate in USD per million tokens is the same number as micro-USD per
token, so "2.5" below means 2.5 micro-USD for each token. A call's cost is the SUM over its token
lines, rounded UP ONCE to a whole micro-USD (not each line on its own).

Rates used (from app/llm/prices.py; USD per million tokens = micro-USD per token):
    claude-haiku-5-5, short prompt: input 0.1, output 0.5, cache read 0.01, write 5m 0.125,
        write 1h 0.2; above 100,000 input tokens: 0.5, 2.5, 0.05, 0.625, 1
    claude-sonnet-5-5: input 2, output 10, cache read 0.1, write 5m 2.5, write 1h 4
    claude-opus-5-5: input 4, output 20, cache read 0.2, write 5m 5, write 1h 8
    openai:text-embedding-3-small: input 0.02, nothing else
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages.ai import UsageMetadata

from app.llm import prices
from app.llm.prices import EMBEDDING_PRICES, PRICES

HAIKU = PRICES["anthropic:claude-haiku-5-5"]
SONNET = PRICES["anthropic:claude-sonnet-5-5"]
OPUS = PRICES["anthropic:claude-opus-5-5"]
EMBED = EMBEDDING_PRICES["openai:text-embedding-3-small"]


def usage(
    input_tokens: int,
    output_tokens: int,
    input_details: dict[str, Any] | None = None,
    output_details: dict[str, Any] | None = None,
) -> UsageMetadata:
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if input_details is not None:
        result["input_token_details"] = input_details
    if output_details is not None:
        result["output_token_details"] = output_details
    return result  # type: ignore[return-value]


# --- rounding: up, once, on the total -----------------------------------------------------------


def test_the_total_is_rounded_up_once_not_each_line() -> None:
    # Haiku. 3 input x 0.1 = 0.3 and 1 output x 0.5 = 0.5; total 0.8 -> 1.
    # Rounding each line would give 1 + 1 = 2.
    assert prices.cost(HAIKU, usage(3, 1)) == 1


def test_two_halves_make_exactly_one_micro_usd() -> None:
    # Haiku. 5 input x 0.1 = 0.5 and 1 output x 0.5 = 0.5; total exactly 1.0 -> 1 (not 2: an
    # exact total is not rounded up again, and per-line rounding would also give 2).
    assert prices.cost(HAIKU, usage(5, 1)) == 1


def test_every_cache_line_is_summed_before_rounding() -> None:
    # Haiku, 10 input of which 4 cache read, 3 written for 5 minutes, 1 written for an hour, so
    # 10 - 4 - 3 - 1 = 2 plain. Lines: 2 x 0.1 = 0.2; 4 x 0.01 = 0.04; 3 x 0.125 = 0.375;
    # 1 x 0.2 = 0.2. Total = 0.2 + 0.04 + 0.375 + 0.2 = 0.815 -> 1.
    # Rounding each of the four lines up would give 1 + 1 + 1 + 1 = 4.
    u = usage(
        10, 0, {"cache_read": 4, "ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 1}
    )
    assert prices.cost(HAIKU, u) == 1


def test_a_total_just_above_a_whole_number_rounds_up() -> None:
    # Sonnet, 1,001 input of which 601 cache read, 100 written 5m, 100 written 1h; plain = 200.
    # 200 x 2 = 400; 601 x 0.1 = 60.1; 100 x 2.5 = 250; 100 x 4 = 400; 200 output x 10 = 2,000.
    # Total = 400 + 60.1 + 250 + 400 + 2,000 = 3,110.1 -> 3,111.
    u = usage(
        1_001,
        200,
        {"cache_read": 601, "ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 100},
    )
    assert prices.cost(SONNET, u) == 3_111


def test_two_fractional_cache_lines_are_summed_before_rounding() -> None:
    # Sonnet, 3 input = 1 cache read + 1 written 5m + 1 plain.
    # 1 x 2 = 2; 1 x 0.1 = 0.1; 1 x 2.5 = 2.5. Total 4.6 -> 5. Rounding each line would give
    # 2 + 1 + 3 = 6.
    assert prices.cost(SONNET, usage(3, 0, {"cache_read": 1, "ephemeral_5m_input_tokens": 1})) == 5


def test_zero_is_zero_and_reasoning_tokens_add_nothing() -> None:
    assert prices.cost(HAIKU, usage(0, 0)) == 0
    # Reasoning tokens are already inside output_tokens: 200 output x 10 = 2,000 with or without.
    assert prices.cost(SONNET, usage(0, 200)) == 2_000
    assert prices.cost(SONNET, usage(0, 200, None, {"reasoning": 150})) == 2_000


# --- cost: how the details are read -------------------------------------------------------------


def test_a_detail_that_is_none_or_missing_counts_as_zero() -> None:
    # Sonnet, 100 input: no details, empty details and None values all give 100 x 2 = 200.
    assert prices.cost(SONNET, usage(100, 0)) == 200
    assert prices.cost(SONNET, usage(100, 0, {})) == 200
    assert prices.cost(SONNET, usage(100, 0, {"cache_read": None, "cache_creation": None})) == 200


def test_details_larger_than_the_input_are_still_charged_and_plain_input_is_never_negative() -> (
    None
):
    # Haiku, 10 input but 20 cache-read tokens reported: plain = max(10 - 20, 0) = 0, and the
    # 20 cache-read tokens are still charged: 20 x 0.01 = 0.2 -> 1. (A provider should never do
    # this; the cost errs upward.)
    assert prices.cost(HAIKU, usage(10, 0, {"cache_read": 20})) == 1


def test_the_long_prompt_tier_is_chosen_by_input_tokens_alone() -> None:
    # Haiku, 100,000 input stays on the short tier even with a large output:
    # 100,000 x 0.1 = 10,000 plus 1,000,000 output x 0.5 = 500,000 -> 510,000.
    assert prices.cost(HAIKU, usage(100_000, 1_000_000)) == 510_000
    # 100,001 input moves the whole call to the long tier: 100,001 x 0.5 = 50,000.5 plus
    # 1,000,000 x 2.5 = 2,500,000 -> 2,550,000.5 -> 2,550,001.
    assert prices.cost(HAIKU, usage(100_001, 1_000_000)) == 2_550_001


def test_cache_lines_on_the_long_prompt_tier_use_the_long_rates() -> None:
    # Haiku, 200,000 input (long tier: input 0.5, read 0.05, write 5m 0.625, write 1h 1):
    # 100,000 read, 20,000 written 5m, 10,000 written 1h; plain = 200,000 - 130,000 = 70,000.
    # 70,000 x 0.5 = 35,000; 100,000 x 0.05 = 5,000; 20,000 x 0.625 = 12,500; 10,000 x 1 = 10,000.
    # Total 62,500 exactly.
    u = usage(
        200_000,
        0,
        {
            "cache_read": 100_000,
            "ephemeral_5m_input_tokens": 20_000,
            "ephemeral_1h_input_tokens": 10_000,
        },
    )
    assert prices.cost(HAIKU, u) == 62_500


def test_an_undivided_cache_write_alongside_a_split_is_charged_on_top() -> None:
    # Opus: 100 input = 10 undivided write (dearer 1h rate 8) + 20 5m write (5) + 70 plain (4).
    # 70 x 4 = 280; 20 x 5 = 100; 10 x 8 = 80. Total 460.
    u = usage(100, 0, {"cache_creation": 10, "ephemeral_5m_input_tokens": 20})
    assert prices.cost(OPUS, u) == 460


def test_a_string_count_is_read_as_a_number() -> None:
    # int(details.get(key) or 0): a provider sending "40" is read as 40. Sonnet: 100 input, 40
    # cache read, 60 plain: 60 x 2 + 40 x 0.1 = 124.
    assert prices.cost(SONNET, usage(100, 0, {"cache_read": "40"})) == 124


def test_embedding_prices_charge_input_only() -> None:
    # 7 tokens x 0.02 = 0.14 -> 1; 50,000 tokens x 0.02 = 1,000 exactly.
    assert prices.cost(EMBED, usage(7, 0)) == 1
    assert prices.cost(EMBED, usage(50_000, 0)) == 1_000


# --- ceiling ------------------------------------------------------------------------------------


def test_ceiling_prices_input_at_the_dearest_of_input_and_long_cache_write() -> None:
    # Opus: max(input 4, long write 8) = 8. 1,000 x 8 = 8,000; 500 output x 20 = 10,000 -> 18,000.
    assert prices.ceiling(OPUS, 1_000, 500) == 18_000
    # Sonnet: max(2, 4) = 4. 1,000 x 4 + 500 x 10 = 4,000 + 5,000 = 9,000.
    assert prices.ceiling(SONNET, 1_000, 500) == 9_000


def test_ceiling_uses_the_long_prompt_rates_even_for_a_short_prompt() -> None:
    # Haiku: highest = long tier (input 0.5, write 1h 1, output 2.5), whatever the prompt length.
    # 10 x max(0.5, 1) = 10; 4 output x 2.5 = 10 -> 20.
    assert prices.ceiling(HAIKU, 10, 4) == 20


def test_ceiling_rounds_up_once_on_the_total() -> None:
    # Haiku, 1 input x 1 = 1 and 1 output x 2.5 = 2.5; total 3.5 -> 4 (per line: 1 + 3 = 4).
    assert prices.ceiling(HAIKU, 1, 1) == 4
    # Embedding model: 1 x max(0.02, 0) = 0.02 -> 1; nothing at all -> 0.
    assert prices.ceiling(EMBED, 1, 0) == 1
    assert prices.ceiling(EMBED, 0, 0) == 0
    # Haiku, 0 input + 1 output x 2.5 = 2.5 -> 3; 1 input x 1 + 0 output = 1 exactly -> 1.
    assert prices.ceiling(HAIKU, 0, 1) == 3
    assert prices.ceiling(HAIKU, 1, 0) == 1


# --- token_details ------------------------------------------------------------------------------


def test_token_details_without_usage_are_none_not_zero() -> None:
    nothing = {"reasoning_tokens": None, "cache_read_tokens": None, "cache_write_tokens": None}
    assert prices.token_details(None) == nothing
    assert prices.token_details({}) == nothing  # type: ignore[arg-type]


def test_token_details_with_usage_but_no_details_record_zero() -> None:
    # Surprising but current: once usage is present, tokens it does not report are recorded as 0
    # (a measured zero), not None (unknown).
    zeros = {"reasoning_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert prices.token_details(usage(10, 5)) == zeros
    assert prices.token_details(usage(10, 5, {}, {})) == zeros


def test_token_details_read_reasoning_cache_reads_and_all_three_write_keys() -> None:
    u = usage(
        1_000,
        300,
        {
            "cache_read": 400,
            "cache_creation": 7,
            "ephemeral_5m_input_tokens": 20,
            "ephemeral_1h_input_tokens": 30,
            "audio": 99,  # not a cache detail: ignored
        },
        {"reasoning": 120, "audio": 5},
    )
    # writes = 7 + 20 + 30 = 57
    assert prices.token_details(u) == {
        "reasoning_tokens": 120,
        "cache_read_tokens": 400,
        "cache_write_tokens": 57,
    }


def test_token_details_treat_none_values_as_zero() -> None:
    u = usage(10, 5, {"cache_read": None, "cache_creation": None}, {"reasoning": None})
    assert prices.token_details(u) == {
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


# --- the price table's shape --------------------------------------------------------------------


def test_the_price_tables_have_the_documented_structure() -> None:
    # Structure only: a project edits the prices themselves.
    assert prices.MICRO_USD == 1_000_000
    assert all(isinstance(rate, int) for p in PRICES.values() for rate in vars_of(p.rates))
    # Haiku is the entry with a long-prompt tier, and the tier starts above 100,000 tokens.
    assert (HAIKU.long_prompt_tokens, HAIKU.long_rates is not None) == (100_000, True)
    assert HAIKU.max_prompt_tokens is None
    assert HAIKU.highest is HAIKU.long_rates
    assert HAIKU.for_prompt(100_000) is HAIKU.rates
    assert HAIKU.for_prompt(100_001) is HAIKU.long_rates
    # Embeddings are billed for input only.
    for price in EMBEDDING_PRICES.values():
        rates = price.rates
        assert (rates.output, rates.cache_read, rates.cache_write_short) == (0, 0, 0)
        assert rates.cache_write_long == 0
        assert rates.input > 0


def vars_of(rates: prices.Rates) -> list[int]:
    return [
        rates.input,
        rates.output,
        rates.cache_read,
        rates.cache_write_short,
        rates.cache_write_long,
    ]


def test_the_listed_providers_and_pricing_pages() -> None:
    assert prices.CLAUDE_PRICING == "https://platform.claude.com/docs/en/about-claude/pricing"
    assert prices.OPENAI_PRICING == "https://developers.openai.com/api/docs/pricing"
    assert {model.split(":", 1)[0] for model in PRICES} == {"anthropic", "openai"}
    assert all(model.startswith("openai:") for model in EMBEDDING_PRICES)
