"""What a recording run (`make record-llm`) needs, kept apart from conftest.py so a test can import
it."""

from __future__ import annotations

from collections.abc import Mapping

from app.llm.core.registry import key_setting


def missing_recording_keys(
    chat_model: str, embedding_model: str, environ: Mapping[str, str]
) -> list[str]:
    """The key variables a recording run lacks: one per provider among the two pinned models, so an
    Anthropic chat model with an OpenAI embedding model needs both. A blank value is missing."""
    names = dict.fromkeys(key_setting(model).upper() for model in (chat_model, embedding_model))
    return [name for name in names if not environ.get(name, "").strip()]
