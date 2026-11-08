"""Moved (#33): GatewayEmbeddings is in app/llm/langchain/embeddings.py, and openai_client in
app/llm/core/openai_embeddings.py (OpenAI's embeddings call, with its error table, is there too);
the budget names this module imported (BudgetExhausted, UserBudgetExhausted, reserve_for, settle,
INPUT_MARGIN_TOKENS) answer here too, from app/llm/core/budget.py and bounds.py. Until 0.8.0 the
old names still answer here, with a DeprecationWarning, and are the same objects as the new ones.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import bounds as _bounds
from app.llm.core import budget as _budget
from app.llm.core import openai_embeddings as _openai_embeddings
from app.llm.langchain import embeddings as _embeddings

if TYPE_CHECKING:
    from app.llm.core.bounds import INPUT_MARGIN_TOKENS as INPUT_MARGIN_TOKENS
    from app.llm.core.budget import BudgetExhausted as BudgetExhausted
    from app.llm.core.budget import UserBudgetExhausted as UserBudgetExhausted
    from app.llm.core.budget import reserve_for as reserve_for
    from app.llm.core.budget import settle as settle
    from app.llm.core.openai_embeddings import openai_client as openai_client
    from app.llm.langchain.embeddings import GatewayEmbeddings as GatewayEmbeddings

__all__ = [
    "INPUT_MARGIN_TOKENS",
    "BudgetExhausted",
    "GatewayEmbeddings",
    "UserBudgetExhausted",
    "openai_client",
    "reserve_for",
    "settle",
]

# The budget names this module used to import and so answered too.
_MOVED = {
    "GatewayEmbeddings": _embeddings,
    "openai_client": _openai_embeddings,
    "INPUT_MARGIN_TOKENS": _bounds,
    **dict.fromkeys(("BudgetExhausted", "UserBudgetExhausted", "reserve_for", "settle"), _budget),
}

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            new = _MOVED[name]
            warnings.warn(
                f"app.llm.embeddings.{name} moved to {new.__name__}.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(new, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
