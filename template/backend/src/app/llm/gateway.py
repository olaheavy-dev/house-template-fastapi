"""Moved (#33): the gateway is in app/llm/langchain/gateway.py (Gateway, get_gateway), and
GatewayDep, the name product code imports, is app.llm.GatewayDep. key_setting and KEY_SETTINGS are
in app/llm/core/registry.py. Until 0.8.0 the old names still answer here, with a
DeprecationWarning, and are the same objects as the new ones.

Patch targets moved with the code: patch app.llm.langchain.gateway.openai_client and
app.llm.langchain.gateway.init_chat_model, not the names on this module.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import app.llm as _llm
from app.llm.core import registry as _registry
from app.llm.langchain import gateway as _gateway

if TYPE_CHECKING:
    from app.llm import GatewayDep as GatewayDep
    from app.llm.core.registry import KEY_SETTINGS as KEY_SETTINGS
    from app.llm.core.registry import key_setting as key_setting
    from app.llm.langchain.gateway import Gateway as Gateway
    from app.llm.langchain.gateway import get_gateway as get_gateway

__all__ = ["KEY_SETTINGS", "Gateway", "GatewayDep", "get_gateway", "key_setting"]

_MOVED = {
    "KEY_SETTINGS": _registry,
    "key_setting": _registry,
    "Gateway": _gateway,
    "get_gateway": _gateway,
    "GatewayDep": _llm,
}

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            new = _MOVED[name]
            warnings.warn(
                f"app.llm.gateway.{name} moved to {new.__name__}.{name}: import it from there "
                "(the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(new, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
