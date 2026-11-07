"""The LangChain adapter's usage conversion (#33): usage_metadata to the core's Usage, read the way
prices.cost and token_details read it before the split."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.core.usage import Usage
from app.llm.langchain.usage import from_langchain


def test_every_detail_lands_in_its_field() -> None:
    usage: Any = {
        "input_tokens": 1_000,
        "output_tokens": 300,
        "total_tokens": 1_300,
        "input_token_details": {
            "cache_read": 400,
            "cache_creation": 7,
            "ephemeral_5m_input_tokens": 20,
            "ephemeral_1h_input_tokens": 30,
            "audio": 99,
        },
        "output_token_details": {"reasoning": 120, "audio": 5},
    }
    assert from_langchain(usage) == Usage(
        input_tokens=1_000,
        output_tokens=300,
        reasoning_tokens=120,
        cache_read_tokens=400,
        cache_write_short_tokens=20,
        cache_write_long_tokens=30,
        cache_write_undivided_tokens=7,
    )


def test_missing_none_and_empty_details_are_zero() -> None:
    zero = Usage(input_tokens=10, output_tokens=5)
    base: dict[str, Any] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    shapes: list[Any] = [
        base,
        {**base, "input_token_details": {}, "output_token_details": {}},
        {**base, "input_token_details": None, "output_token_details": None},
        {**base, "input_token_details": {"cache_read": None}, "output_token_details": {}},
    ]
    for shape in shapes:
        assert from_langchain(shape) == zero, shape


def test_a_count_sent_as_a_string_is_read_as_a_number() -> None:
    usage: Any = {
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
        "input_token_details": {"cache_read": "40"},
    }
    assert from_langchain(usage).cache_read_tokens == 40


def test_a_call_without_usage_is_the_callers_to_handle() -> None:
    # The gateway passes None (or LangChain's empty dict) as "no usage" and never converts it.
    with pytest.raises(KeyError):
        from_langchain({})  # type: ignore[typeddict-item]
