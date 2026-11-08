"""Moved (#33): the llm.* OpenTelemetry instruments and record_call are in app/llm/core/metrics.py.
Until 0.8.0 the old names still answer here, with a DeprecationWarning, and are the same objects
as the core's (the instruments are created once, there).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import metrics as _core

if TYPE_CHECKING:
    from app.llm.core.metrics import calls as calls
    from app.llm.core.metrics import cost as cost
    from app.llm.core.metrics import labels as labels
    from app.llm.core.metrics import latency as latency
    from app.llm.core.metrics import record_call as record_call
    from app.llm.core.metrics import record_refusal as record_refusal
    from app.llm.core.metrics import tokens as tokens

__all__ = ["calls", "cost", "labels", "latency", "record_call", "record_refusal", "tokens"]

_MOVED = frozenset(__all__)

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            takes_usage = (
                " and now takes app.llm.core.usage.Usage: convert LangChain usage with "
                "app.llm.langchain.usage.from_langchain (pass None, not {}, for no usage)"
                if name == "record_call"
                else ""
            )
            warnings.warn(
                f"app.llm.metrics.{name} moved to app.llm.core.metrics.{name}: import it from "
                f"there (the old name goes in 0.8.0){takes_usage}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(_core, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
