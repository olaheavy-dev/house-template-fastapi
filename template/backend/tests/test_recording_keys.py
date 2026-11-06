"""The keys a recording run needs: one per provider among the pinned models."""

from __future__ import annotations

from tests.recording import missing_recording_keys


def test_a_recording_with_both_models_from_one_provider_needs_that_key() -> None:
    assert missing_recording_keys("openai:a", "openai:b", {}) == ["OPENAI_API_KEY"]
    assert missing_recording_keys("openai:a", "openai:b", {"OPENAI_API_KEY": "k"}) == []


def test_a_recording_with_two_providers_needs_both_keys() -> None:
    # An Anthropic chat model plus an OpenAI embedding model: the OpenAI key must not be skipped,
    # or the placeholder "test-key" would record the provider's 401 into the cassette.
    env = {"ANTHROPIC_API_KEY": "k"}
    assert missing_recording_keys("anthropic:a", "openai:b", env) == ["OPENAI_API_KEY"]
    assert missing_recording_keys("anthropic:a", "openai:b", {}) == [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]
    both = {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}
    assert missing_recording_keys("anthropic:a", "openai:b", both) == []


def test_an_empty_or_blank_key_counts_as_missing() -> None:
    assert missing_recording_keys("openai:a", "openai:b", {"OPENAI_API_KEY": "  \n"}) == [
        "OPENAI_API_KEY"
    ]
    assert missing_recording_keys("openai:a", "openai:b", {"OPENAI_API_KEY": ""}) == [
        "OPENAI_API_KEY"
    ]
