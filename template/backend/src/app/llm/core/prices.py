"""How a call is priced: integer micro-USD rates per million tokens, the cost of a call from its
Usage, and the most a call can cost before it is sent. The price tables themselves are the
project's, in app/llm/prices.py.

Integers keep the budget arithmetic exact. A call's cost is the sum over its token lines, rounded
UP once to a whole micro-USD, so the budget can over-record a call by less than one micro-USD and
never under-record it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from app.llm.core.usage import Usage

MICRO_USD = 1_000_000
_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Rates:
    input: int
    output: int
    cache_read: int
    # The cache-write rates of the shorter and the longer lifetime: Anthropic's pricing page lists
    # 5-minute and 1-hour writes, OpenAI's lists 30-minute and 12-hour writes.
    cache_write_short: int
    cache_write_long: int


@dataclass(frozen=True, slots=True)
class Price:
    rates: Rates
    # When the price was read (ISO date, YYYY-MM-DD) and the provider's pricing page it came from.
    checked: str = field(kw_only=True)
    source: str = field(kw_only=True)
    # Some models charge more for every token of a request whose prompt is longer than this.
    long_prompt_tokens: int | None = None
    long_rates: Rates | None = None
    # Longest prompt these rates cover, when longer ones cost more and aren't priced here: the
    # budget refuses them instead of under-recording their cost. For an embedding price it bounds
    # each request: texts are batched under it.
    max_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        # Not compared with today's date: a clock that is a day off must not stop the app starting
        # (a test checks that no entry is dated after tomorrow, UTC: people east of UTC are a day
        # ahead for part of the day).
        try:
            exact = date.fromisoformat(self.checked).isoformat() == self.checked
        except ValueError:
            exact = False
        if not exact:
            raise ValueError(f"Price.checked must be a date as YYYY-MM-DD, not {self.checked!r}")
        try:
            url = urlparse(self.source)
            valid_url = url.scheme == "https" and bool(url.hostname)
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(
                f"Price.source must be the pricing page's https URL, not {self.source!r}"
            )

    def for_prompt(self, input_tokens: int) -> Rates:
        if self.long_rates and self.long_prompt_tokens and input_tokens > self.long_prompt_tokens:
            return self.long_rates
        return self.rates

    @property
    def highest(self) -> Rates:
        return self.long_rates or self.rates


def usd_rates(input: float, output: float, cache_read: float, short: float, long: float) -> Rates:
    """Rates from a pricing page's USD per million tokens: input, output, cache read, cache write
    (shorter lifetime), cache write (longer lifetime)."""

    def micro(usd: float) -> int:
        return round(usd * MICRO_USD)

    return Rates(micro(input), micro(output), micro(cache_read), micro(short), micro(long))


def cost(price: Price, usage: Usage) -> int:
    """The cost of one call in micro-USD, rounded up once on the total.

    Cached tokens are part of input_tokens; what is left after the cache lines is plain input. A
    cache write with no lifetime is charged at the longer lifetime's rate, so the cost can be
    over-recorded, never under-recorded. Tokens in a detail Usage doesn't hold (a service tier,
    audio) are charged as plain input or output, which never undercharges a standard-tier call.
    """
    rates = price.for_prompt(usage.input_tokens)
    cached = (
        usage.cache_read_tokens
        + usage.cache_write_short_tokens
        + usage.cache_write_long_tokens
        + usage.cache_write_undivided_tokens
    )
    plain = max(usage.input_tokens - cached, 0)
    total = (
        plain * rates.input
        + usage.cache_read_tokens * rates.cache_read
        + usage.cache_write_short_tokens * rates.cache_write_short
        + (usage.cache_write_long_tokens + usage.cache_write_undivided_tokens)
        * rates.cache_write_long
        + usage.output_tokens * rates.output
    )
    return -(-total // _PER_MILLION)


def ceiling(price: Price, input_tokens: int, output_tokens: int) -> int:
    """The most a call can cost: every input token uncached at the highest rate, every output token
    the call may generate. The budget reserves this before the call."""
    rates = price.highest
    total = input_tokens * max(rates.input, rates.cache_write_long) + output_tokens * rates.output
    return -(-total // _PER_MILLION)


def token_details(usage: Usage | None) -> dict[str, int | None]:
    """The llm_call columns for reasoning, cache-read and cache-write tokens: None without usage,
    otherwise the counts (0 for a detail the provider didn't report, see Usage)."""
    if usage is None:
        return {"reasoning_tokens": None, "cache_read_tokens": None, "cache_write_tokens": None}
    return {
        "reasoning_tokens": usage.reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }
