"""Moved (#33): the failure kinds, outcomes and the neutral handlers are in app/llm/core/errors.py,
LangChain's classification and handlers in app/llm/langchain/errors.py. Until 0.8.0 the old names
still answer here, with a DeprecationWarning, and are the same objects as the new ones.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import errors as _core
from app.llm.langchain import errors as _langchain

if TYPE_CHECKING:
    from app.llm.core.errors import DEFAULT_RETRY_AFTER as DEFAULT_RETRY_AFTER
    from app.llm.core.errors import Outcome as Outcome
    from app.llm.langchain.errors import OUTCOMES as OUTCOMES
    from app.llm.langchain.errors import (
        install_model_error_handlers as install_model_error_handlers,
    )
    from app.llm.langchain.errors import outcome_for as outcome_for

__all__ = [
    "DEFAULT_RETRY_AFTER",
    "OUTCOMES",
    "Outcome",
    "install_model_error_handlers",
    "outcome_for",
]

_MOVED = {
    "DEFAULT_RETRY_AFTER": _core,
    "Outcome": _core,
    "OUTCOMES": _langchain,
    "install_model_error_handlers": _langchain,
    "outcome_for": _langchain,
}

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            new = _MOVED[name]
            warnings.warn(
                f"app.llm.errors.{name} moved to {new.__name__}.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(new, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
