"""Prompts are versioned files in the repo, never pulled at runtime, so a prompt change shows up in
review and a replayed test sends exactly what was recorded.

Each prompt is a Markdown file in app/prompts/ whose content hash is its version. Pass
prompt.config() with the call: the hash goes on the call's cost row and on its LangSmith trace,
so experiments and costs can be compared prompt version by prompt version.

    summarise = load_prompt("summarise")
    answer = await model.ainvoke(
        [SystemMessage(summarise.text), HumanMessage(text)], config=summarise.config()
    )
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from langchain_core.runnables import RunnableConfig

PROMPTS = Path(__file__).parents[1] / "prompts"


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    text: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]

    def config(self) -> RunnableConfig:
        return {
            "run_name": self.name,
            "metadata": {"prompt": self.name, "prompt_sha": self.sha},
            "tags": [f"prompt:{self.name}@{self.sha}"],
        }


@cache
def load_prompt(name: str, directory: Path = PROMPTS) -> Prompt:
    path = (directory / f"{name}.md").resolve()
    if path.parent != directory.resolve():
        raise ValueError(f"not a prompt name: {name!r}")
    return Prompt(name=name, text=path.read_text(encoding="utf-8"))
