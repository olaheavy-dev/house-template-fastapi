"""A prompt for a MAF call: the file, its version, and how to tag a call with it.

    summarise = load_prompt("summarise")
    with gateway.call_options(user, "summarise", summarise):
        answer = await gateway.client().get_response(
            [Message("system", [summarise.text]), Message("user", [text])]
        )

The prompt, its hash and loading the file are in app/llm/core/prompts.py; MAF needs no adapter
class, because the tags travel in `Gateway.call_options` (a context variable).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from app.llm.core import prompts as _core
from app.llm.core.prompts import PROMPTS, Prompt

__all__ = ["Prompt", "load_prompt"]


@cache
def load_prompt(name: str, directory: Path = PROMPTS) -> Prompt:
    return Prompt(name=name, text=_core.prompt_text(name, directory))
