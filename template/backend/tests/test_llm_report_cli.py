"""The spend report's command line (`make llm-report`, a cron's `--expire-stale`): what it prints
and returns for a bad or a help request. None of it needs a database."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

import pytest

from app.llm.core import report
from tests.test_llm_core_embeddings import loaded_by

MODULE = "app.llm.core.report"


def run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    # NO_COLOR: Python 3.14's argparse colours its output on a terminal and with FORCE_COLOR.
    env = {k: v for k, v in os.environ.items() if k != "FORCE_COLOR"} | {"NO_COLOR": "1"}
    return subprocess.run(  # noqa: S603 - this interpreter, running a module of this project
        [sys.executable, "-m", module, *args], capture_output=True, text=True, env=env
    )


def test_an_unknown_group_exits_2_and_names_the_groups() -> None:
    result = run(MODULE, "--by", "nope")
    assert result.returncode == 2
    assert result.stderr.endswith("error: --by takes day, model, prompt, feature\n")
    assert result.stdout == ""


def test_help_exits_0_and_describes_the_report() -> None:
    result = run(MODULE, "--help")
    assert result.returncode == 0
    assert "Model spend from the llm_call table." in result.stdout


# --- the arguments reach main() ---------------------------------------------------------------


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, list[str], bool]]:
    calls: list[tuple[int, list[str], bool]] = []

    async def fake(days: int, by: Sequence[str], expire: bool) -> None:
        calls.append((days, list(by), expire))

    monkeypatch.setattr(report, "main", fake)
    return calls


def test_the_defaults_are_a_week_by_day_and_model(seen: list[tuple[int, list[str], bool]]) -> None:
    report.cli([])
    assert seen == [(7, ["day", "model"], False)]


def test_by_is_split_on_commas_and_stripped(seen: list[tuple[int, list[str], bool]]) -> None:
    report.cli(["--days", "30", "--by", " feature , model "])
    assert seen == [(30, ["feature", "model"], False)]


def test_expire_stale_is_passed_on(seen: list[tuple[int, list[str], bool]]) -> None:
    report.cli(["--expire-stale"])
    assert seen == [(7, ["day", "model"], True)]


@pytest.mark.parametrize("by", [",", " ", "day,nope"])
def test_a_bad_by_is_refused_before_main_runs(
    by: str, seen: list[tuple[int, list[str], bool]]
) -> None:
    with pytest.raises(SystemExit) as stop:
        report.cli(["--by", by])
    assert stop.value.code == 2
    assert seen == []


def test_importing_the_report_loads_no_model_framework() -> None:
    assert loaded_by("app.llm.core.report") == "[]"
