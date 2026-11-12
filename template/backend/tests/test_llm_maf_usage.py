"""MAF's usage_details as the core's Usage, and what that costs (#35). Written before the code.

The expected costs are worked out by hand from the price tables and the rule in
app/llm/core/prices.py cost(): plain input (input minus every cached line), cache reads, cache
writes and output are each multiplied by the model's micro-USD rate per million tokens, the sum is
divided by 1,000,000 and rounded UP once. The working is beside each test, so a wrong
implementation cannot confirm itself.

MAF (agent-framework-core 1.25.0) reports, in UsageDetails: input_token_count (the total prompt
tokens, cached ones included), output_token_count (reasoning included),
reasoning_output_token_count, cache_read_input_token_count and cache_creation_input_token_count.
The OpenAI Responses client puts input_tokens, output_tokens and the cached and reasoning details
there; the Anthropic client adds cache_creation and cache_read to the input total
(agent_framework_anthropic/_chat_client.py _parse_usage_from_anthropic), and never says how long
a cache write lives.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.llm import prices
from app.llm.core.prices import cost
from app.llm.core.usage import Usage
from app.llm.maf.usage import from_maf

# claude-sonnet-5-5, USD per million tokens: input 2, output 10, cache read 0.10, cache write 2.50
# (5 minutes) and 4 (1 hour). In micro-USD per million: 2,000,000; 10,000,000; 100,000; 2,500,000;
# 4,000,000.
SONNET = prices.PRICES["anthropic:claude-sonnet-5-5"]
# gpt-6-luna: input 0.10, output 0.50, cache read 0.01 -> 100,000; 500,000; 10,000.
LUNA = prices.PRICES["openai:gpt-6-luna"]


def test_every_key_lands_in_its_field() -> None:
    details: Any = {
        "input_token_count": 1_000,
        "output_token_count": 300,
        "total_token_count": 1_300,
        "reasoning_output_token_count": 120,
        "cache_read_input_token_count": 400,
        "cache_creation_input_token_count": 7,
        "openai.cached_input_tokens": 400,
    }
    assert from_maf(details) == Usage(
        input_tokens=1_000,
        output_tokens=300,
        reasoning_tokens=120,
        cache_read_tokens=400,
        cache_write_undivided_tokens=7,
    )


def test_an_openai_call_is_costed_with_its_cached_input_at_the_cache_rate() -> None:
    # 1,000 input tokens of which 400 were cached; 300 output tokens of which 120 were reasoning
    # (the output count already includes them).
    usage = from_maf(
        {
            "input_token_count": 1_000,
            "output_token_count": 300,
            "reasoning_output_token_count": 120,
            "cache_read_input_token_count": 400,
        }
    )
    assert usage is not None
    # plain input   (1,000 - 400) x 100,000   =  60,000,000
    # cache read          400 x  10,000        =   4,000,000
    # output              300 x 500,000        = 150,000,000
    # total 214,000,000 / 1,000,000 = 214 micro-USD exactly.
    assert cost(LUNA, usage) == 214


def test_an_anthropic_cache_write_is_charged_at_the_longer_lifetimes_rate() -> None:
    # The API said: 100 input tokens, 200 written to the cache, 300 read from it, 50 output. MAF
    # adds the cache lines to the input: 100 + 200 + 300 = 600.
    usage = from_maf(
        {
            "input_token_count": 600,
            "output_token_count": 50,
            "cache_creation_input_token_count": 200,
            "cache_read_input_token_count": 300,
        }
    )
    assert usage == Usage(
        input_tokens=600, output_tokens=50, cache_read_tokens=300, cache_write_undivided_tokens=200
    )
    assert usage is not None
    # plain input   (600 - 300 - 200) x  2,000,000 =   200,000,000
    # cache read                 300 x    100,000 =    30,000,000
    # cache write, longer rate   200 x  4,000,000 =   800,000,000
    # output                      50 x 10,000,000 =   500,000,000
    # total 1,530,000,000 / 1,000,000 = 1,530 micro-USD. At the shorter rate (2,500,000) the write
    # would be 500,000,000 and the total 1,230: the longer rate is a deliberate, safe over-count,
    # because MAF drops the lifetime the API reports.
    assert cost(SONNET, usage) == 1_530


def test_a_call_with_nothing_but_the_two_totals_has_zero_details() -> None:
    usage = from_maf({"input_token_count": 10, "output_token_count": 5})
    assert usage == Usage(input_tokens=10, output_tokens=5)
    # 10 x 2,000,000 + 5 x 10,000,000 = 70,000,000 -> 70 micro-USD.
    assert usage is not None
    assert cost(SONNET, usage) == 70


@pytest.mark.parametrize(
    "details",
    [
        None,
        {},
        {"input_token_count": 10},
        {"output_token_count": 5},
        {"input_token_count": None, "output_token_count": 5},
        {"input_token_count": 10, "output_token_count": None},
        {"total_token_count": 15},
    ],
)
def test_usage_that_is_missing_or_incomplete_is_unknown_never_zero(details: Any) -> None:
    # The budget then keeps the reservation as the cost (an over-count), instead of recording a
    # cheap call that may not have been.
    assert from_maf(details) is None


@pytest.mark.parametrize(
    "bad",
    [-1, True, 1.5, "40", [1], {"a": 1}],
)
def test_a_count_that_is_not_a_whole_non_negative_number_makes_the_usage_unknown(bad: Any) -> None:
    base: dict[str, Any] = {"input_token_count": 10, "output_token_count": 5}
    for key in (
        "input_token_count",
        "output_token_count",
        "reasoning_output_token_count",
        "cache_read_input_token_count",
        "cache_creation_input_token_count",
    ):
        assert from_maf({**base, key: bad}) is None, key


def test_a_detail_that_is_none_counts_as_zero() -> None:
    usage = from_maf(
        {
            "input_token_count": 10,
            "output_token_count": 5,
            "cache_read_input_token_count": None,
            "reasoning_output_token_count": None,
            "cache_creation_input_token_count": None,
        }
    )
    assert usage == Usage(input_tokens=10, output_tokens=5)


def test_a_cache_larger_than_the_input_is_still_never_a_negative_cost() -> None:
    # A provider bug, not a case to trust: cost() floors plain input at zero, so the call is
    # priced on its cache lines and output, never below.
    usage = from_maf(
        {
            "input_token_count": 100,
            "output_token_count": 0,
            "cache_read_input_token_count": 150,
        }
    )
    assert usage is not None
    # plain input max(100 - 150, 0) = 0; cache read 150 x 100,000 = 15,000,000 -> 15 micro-USD.
    assert cost(SONNET, usage) == 15
