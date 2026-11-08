"""Moved (#33): the ledger, the Meter and the budget's errors are in app/llm/core/budget.py, the
request bounds in app/llm/core/bounds.py, and LangChain's BudgetGuard and input_bound in
app/llm/langchain/callbacks.py. Until 0.8.0 the old names still answer here, with a
DeprecationWarning, and are the same objects as the new ones (the llm_call tables are defined once,
in app/llm/core/budget.py).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from app.llm.core import bounds as _bounds
from app.llm.core import budget as _budget
from app.llm.langchain import callbacks as _callbacks

if TYPE_CHECKING:
    from app.llm.core.bounds import IMAGE_TOKENS as IMAGE_TOKENS
    from app.llm.core.bounds import INPUT_MARGIN_TOKENS as INPUT_MARGIN_TOKENS
    from app.llm.core.bounds import PromptTooLong as PromptTooLong
    from app.llm.core.bounds import UnboundedMedia as UnboundedMedia
    from app.llm.core.bounds import UnpricedRequest as UnpricedRequest
    from app.llm.core.bounds import check_priced as check_priced
    from app.llm.core.budget import WARN_SHARE as WARN_SHARE
    from app.llm.core.budget import BudgetExhausted as BudgetExhausted
    from app.llm.core.budget import LlmCall as LlmCall
    from app.llm.core.budget import LlmSpendDay as LlmSpendDay
    from app.llm.core.budget import LlmSpendUserDay as LlmSpendUserDay
    from app.llm.core.budget import UserBudgetExhausted as UserBudgetExhausted
    from app.llm.core.budget import expire_stale as expire_stale
    from app.llm.core.budget import reserve as reserve
    from app.llm.core.budget import reserve_for as reserve_for
    from app.llm.core.budget import settle as settle
    from app.llm.core.budget import utc_today as utc_today
    from app.llm.langchain.callbacks import BudgetGuard as BudgetGuard
    from app.llm.langchain.callbacks import input_bound as input_bound

__all__ = [
    "IMAGE_TOKENS",
    "INPUT_MARGIN_TOKENS",
    "WARN_SHARE",
    "BudgetExhausted",
    "BudgetGuard",
    "LlmCall",
    "LlmSpendDay",
    "LlmSpendUserDay",
    "PromptTooLong",
    "UnboundedMedia",
    "UnpricedRequest",
    "UserBudgetExhausted",
    "check_priced",
    "expire_stale",
    "input_bound",
    "reserve",
    "reserve_for",
    "settle",
    "utc_today",
]

_MOVED = {
    **dict.fromkeys(
        (
            "IMAGE_TOKENS",
            "INPUT_MARGIN_TOKENS",
            "PromptTooLong",
            "UnboundedMedia",
            "UnpricedRequest",
            "check_priced",
        ),
        _bounds,
    ),
    **dict.fromkeys(
        (
            "WARN_SHARE",
            "BudgetExhausted",
            "LlmCall",
            "LlmSpendDay",
            "LlmSpendUserDay",
            "UserBudgetExhausted",
            "expire_stale",
            "reserve",
            "reserve_for",
            "settle",
            "utc_today",
        ),
        _budget,
    ),
    **dict.fromkeys(("BudgetGuard", "input_bound"), _callbacks),
}
# Its signature changed too: usage= is the core's Usage now, not LangChain's usage_metadata.
_TAKES_USAGE = frozenset({"settle"})

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            new = _MOVED[name]
            takes_usage = (
                " and now takes app.llm.core.usage.Usage: convert LangChain usage with "
                "app.llm.langchain.usage.from_langchain (pass None, not {}, for no usage)"
                if name in _TAKES_USAGE
                else ""
            )
            warnings.warn(
                f"app.llm.budget.{name} moved to {new.__name__}.{name}: import it from there "
                f"(the old name goes in 0.8.0){takes_usage}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(new, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
