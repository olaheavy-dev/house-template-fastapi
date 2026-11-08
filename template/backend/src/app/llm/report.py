"""Moved (#33): the spend report is in app/llm/core/report.py. Until 0.8.0 the old names still
answer here, with a DeprecationWarning, and are the same objects as the new ones, and
`python -m app.llm.report` still runs the report, with a DeprecationWarning: point a scheduled job
at `python -m app.llm.core.report` now.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import report as _core

if TYPE_CHECKING:
    from app.llm.core.report import GROUPS as GROUPS
    from app.llm.core.report import STALE_AFTER as STALE_AFTER
    from app.llm.core.report import SpendRow as SpendRow
    from app.llm.core.report import collect as collect
    from app.llm.core.report import main as main
    from app.llm.core.report import render as render
    from app.llm.core.report import spend as spend
    from app.llm.core.report import stale as stale

__all__ = ["GROUPS", "STALE_AFTER", "SpendRow", "collect", "main", "render", "spend", "stale"]

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in __all__:
            warnings.warn(
                f"app.llm.report.{name} moved to {_core.__name__}.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(_core, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # stacklevel=1: the warning names this module, so the default filter shows it under `-m`.
    warnings.warn(
        "python -m app.llm.report moved to python -m app.llm.core.report: run that instead "
        "(the old one goes in 0.8.0)",
        DeprecationWarning,
        stacklevel=1,
    )
    _core.cli()
