"""What one model call used, in tokens, whatever framework made the call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Token counts as the provider reported them. input_tokens includes every cached token (read
    or written), and output_tokens includes reasoning tokens, as both providers count them.

    The details are 0 when the provider reported zero AND when it reported nothing: the budget has
    always recorded an unreported detail as 0, never as unknown, and llm_call keeps that meaning.
    A call with no usage at all is None where a Usage is expected, never Usage(0, 0).
    """

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    # Cache writes by lifetime: the shorter one (Anthropic 5 minutes, OpenAI 30 minutes), the longer
    # one (1 hour, 12 hours), and a count reported with no lifetime, which is charged at the longer
    # lifetime's (dearer) rate.
    cache_write_short_tokens: int = 0
    cache_write_long_tokens: int = 0
    cache_write_undivided_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return (
            self.cache_write_short_tokens
            + self.cache_write_long_tokens
            + self.cache_write_undivided_tokens
        )
