"""The agent's model-call limit setting (#34): AGENT_MAX_MODEL_CALLS replaces AGENT_MAX_STEPS, and
the app refuses to start while the old name is set, because the unit changed (LangGraph steps to
model calls) and keeping the number would double the worst-case spend."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_the_default_is_13_model_calls() -> None:
    assert Settings(_env_file=None).agent_max_model_calls == 13


def test_a_limit_of_one_is_allowed_and_zero_is_not() -> None:
    assert Settings(_env_file=None, agent_max_model_calls=1).agent_max_model_calls == 1
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_max_model_calls=0)


def test_the_environment_sets_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "5")
    assert Settings(_env_file=None).agent_max_model_calls == 5


def test_the_old_environment_variable_stops_the_app_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "25")
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)
    message = str(caught.value)
    assert "AGENT_MAX_STEPS" in message
    assert "AGENT_MAX_MODEL_CALLS" in message
    # The old value V allowed ceil(V / 2) calls: 25 steps allowed 13.
    assert "ceil" in message


def test_the_old_name_in_the_env_file_stops_the_app_starting(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MAX_STEPS=25\n")
    with pytest.raises(ValidationError, match="AGENT_MAX_STEPS"):
        Settings(_env_file=env_file)


def test_the_old_name_given_directly_stops_the_app_starting() -> None:
    with pytest.raises(ValidationError, match="AGENT_MAX_STEPS"):
        Settings(_env_file=None, agent_max_steps=25)


def test_the_new_name_in_the_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MAX_MODEL_CALLS=7\n")
    assert Settings(_env_file=env_file).agent_max_model_calls == 7


@pytest.mark.parametrize("name", ["agent_max_steps", "Agent_Max_Steps", "AGENT_max_STEPS"])
def test_the_old_environment_variable_in_any_case_stops_the_app_starting(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # Settings reads environment names case-insensitively, so a lower-case agent_max_steps=5 used
    # to be honoured on Linux; it must be refused too, not ignored.
    monkeypatch.setenv(name, "5")
    with pytest.raises(ValidationError, match="AGENT_MAX_STEPS"):
        Settings(_env_file=None)


@pytest.mark.parametrize("key", ["agent_max_steps", "Agent_Max_Steps", "AGENT_MAX_STEPS"])
def test_the_old_name_in_the_env_file_in_any_case_stops_the_app_starting(
    tmp_path: Path, key: str
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}=5\n")
    with pytest.raises(ValidationError, match="AGENT_MAX_STEPS"):
        Settings(_env_file=env_file)


# --- personal-data redaction settings (#78) --------------------------------------------------


def test_redaction_is_on_by_default_with_the_documented_kinds() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_redaction == "on"
    assert settings.llm_redact_kinds == "EMAIL,PHONE,NINO,IBAN,SORT_CODE,ACCOUNT,CARD,NHS,DOB"


def test_the_environment_turns_redaction_off_and_narrows_the_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_REDACTION", "off")
    monkeypatch.setenv("LLM_REDACT_KINDS", "EMAIL,PHONE")
    settings = Settings(_env_file=None)
    assert (settings.llm_redaction, settings.llm_redact_kinds) == ("off", "EMAIL,PHONE")


@pytest.mark.parametrize("value", ["maybe", "ON ", "true", ""])
def test_redaction_accepts_only_on_or_off(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_redaction=value)
