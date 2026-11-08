"""Moved (#33): the prompt, its hash and loading the file are in app/llm/core/prompts.py (PROMPTS),
and the LangChain `Prompt` with `config()` and `load_prompt` in app/llm/langchain/prompts.py. Until
0.8.0 the old names still answer here, with a DeprecationWarning, and are the same objects as the
new ones.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import prompts as _core
from app.llm.langchain import prompts as _langchain

if TYPE_CHECKING:
    from app.llm.core.prompts import PROMPTS as PROMPTS
    from app.llm.langchain.prompts import Prompt as Prompt
    from app.llm.langchain.prompts import load_prompt as load_prompt

__all__ = ["PROMPTS", "Prompt", "load_prompt"]

_MOVED = {"PROMPTS": _core, "Prompt": _langchain, "load_prompt": _langchain}

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            new = _MOVED[name]
            warnings.warn(
                f"app.llm.prompts.{name} moved to {new.__name__}.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(new, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
