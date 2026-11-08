"""LangChain's side of a prompt: the call config that carries its version.

    summarise = load_prompt("summarise")
    answer = await model.ainvoke(
        [SystemMessage(summarise.text), HumanMessage(text)], config=summarise.config()
    )

The prompt, its hash and loading the file are in app/llm/core/prompts.py. `Prompt` here is that
class plus `config()`, so existing `prompt.config()` calls keep working.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from app.llm.core import prompts as _core
from app.llm.core.prompts import PROMPTS


def prompt_config(prompt: _core.Prompt) -> RunnableConfig:
    """The config to pass with a call: the prompt's name, its version in the metadata (which the
    cost row and the trace record) and its tag."""
    return {"run_name": prompt.name, "metadata": prompt.metadata(), "tags": [prompt.tag]}


class Prompt(_core.Prompt):
    __slots__ = ()

    def config(self) -> RunnableConfig:
        return prompt_config(self)


@cache
def load_prompt(name: str, directory: Path = PROMPTS) -> Prompt:
    return Prompt(name=name, text=_core.prompt_text(name, directory))
