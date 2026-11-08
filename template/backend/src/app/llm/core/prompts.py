"""Prompts are versioned files in the repo, never pulled at runtime, so a prompt change shows up in
review and a replayed test sends exactly what was recorded.

Each prompt is a Markdown file in app/prompts/ whose content hash is its version. The hash goes on
the call's cost row (`metadata()`) and, with `tag`, on its trace, so experiments and costs can be
compared prompt version by prompt version. This module knows no model framework: a framework's
adapter turns a prompt into its own call config (app/llm/langchain/prompts.py for LangChain).

A core Prompt and an adapter's Prompt with equal fields are different types, so they do not compare
equal; compare their `sha`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPTS = Path(__file__).parents[2] / "prompts"


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    text: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]

    @property
    def tag(self) -> str:
        return f"prompt:{self.name}@{self.sha}"

    def metadata(self) -> dict[str, str]:
        return {"prompt": self.name, "prompt_sha": self.sha}


def prompt_text(name: str, directory: Path = PROMPTS) -> str:
    """The text of `directory/<name>.md`, exactly as written (nothing stripped). A name that is
    not a plain file name in `directory` (`sub/x`, `../x`) is refused."""
    path = (directory / f"{name}.md").resolve()
    if path.parent != directory.resolve():
        raise ValueError(f"not a prompt name: {name!r}")
    return path.read_text(encoding="utf-8")
