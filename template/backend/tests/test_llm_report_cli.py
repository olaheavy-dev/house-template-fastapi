"""The spend report's command line (`make llm-report`, a cron's `--expire-stale`): what it prints
and returns for a bad or a help request. Neither needs a database. Pinned (#33) before the report
moves, so the move keeps the command working."""

from __future__ import annotations

import os
import subprocess
import sys

MODULE = "app.llm.report"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    # NO_COLOR: Python 3.14's argparse colours its output on a terminal and with FORCE_COLOR.
    env = {k: v for k, v in os.environ.items() if k != "FORCE_COLOR"} | {"NO_COLOR": "1"}
    return subprocess.run(  # noqa: S603 - this interpreter, running a module of this project
        [sys.executable, "-m", MODULE, *args], capture_output=True, text=True, env=env
    )


def test_an_unknown_group_exits_2_and_names_the_groups() -> None:
    result = run("--by", "nope")
    assert result.returncode == 2
    assert result.stderr.endswith("error: --by takes day, model, prompt, feature\n")
    assert result.stdout == ""


def test_help_exits_0_and_describes_the_report() -> None:
    result = run("--help")
    assert result.returncode == 0
    assert "Model spend from the llm_call table." in result.stdout
