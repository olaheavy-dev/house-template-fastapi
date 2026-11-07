"""Which setting holds a provider's key, and the price of the model a setting names."""

from __future__ import annotations

from collections.abc import Mapping

from app.llm.core.prices import Price

# The setting (and environment variable, upper-cased) that holds each provider's key.
KEY_SETTINGS = {"anthropic": "anthropic_api_key", "openai": "openai_api_key"}


def key_setting(model: str) -> str:
    """The setting holding the key for LLM_MODEL's provider, e.g. "openai_api_key"."""
    provider = model.split(":", 1)[0]
    if provider not in KEY_SETTINGS:
        raise ValueError(
            f"LLM_MODEL {model!r} names no supported provider ({sorted(KEY_SETTINGS)})"
        )
    return KEY_SETTINGS[provider]


def chat_price(model: str, prices: Mapping[str, Price]) -> Price:
    """LLM_MODEL's entry in the chat price table (app/llm/prices.py PRICES)."""
    price = prices.get(model)
    if price is None:
        raise ValueError(f"LLM_MODEL {model!r} has no price in app/llm/prices.py")
    return price


def embedding_price(model: str, prices: Mapping[str, Price]) -> Price:
    """EMBEDDING_MODEL's entry in the embedding price table (app/llm/prices.py EMBEDDING_PRICES)."""
    price = prices.get(model)
    if price is None:
        raise ValueError(
            f"EMBEDDING_MODEL {model!r} has no price in app/llm/prices.py EMBEDDING_PRICES"
        )
    return price
