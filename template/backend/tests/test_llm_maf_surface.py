"""Where MAF may be imported (#35): only app/llm/maf, and only there may a MAF client be built.
The import contract (lint-imports) says the first; this reads the syntax tree for both, so a client
built under another name or a lazy import is caught too."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "app"
MAF = SRC / "llm" / "maf"
CLIENTS = {
    "OpenAIChatClient",
    "OpenAIChatCompletionClient",
    "RawOpenAIChatClient",
    "AnthropicClient",
    "RawAnthropicClient",
    "BaseChatClient",
}


def imported(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names


def test_agent_framework_is_imported_only_under_app_llm_maf() -> None:
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if MAF not in path.parents
        and any(name.split(".")[0].startswith("agent_framework") for name in imported(path))
    ]
    assert offenders == []


def test_a_maf_client_is_built_only_by_the_gateway() -> None:
    offenders: list[str] = []
    for path in MAF.glob("*.py"):
        if path.name == "gateway.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            name = node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)
            if name in CLIENTS:
                offenders.append(f"{path.name}:{node.lineno}")  # type: ignore[attr-defined]
    assert offenders == []


def test_the_maf_adapter_imports_no_langchain() -> None:
    for path in MAF.glob("*.py"):
        assert not [n for n in imported(path) if n.split(".")[0].startswith("lang")], path.name


def test_the_core_imports_no_maf_adapter() -> None:
    for path in (SRC / "llm" / "core").glob("*.py"):
        assert not [n for n in imported(path) if n.startswith("app.llm.maf")], path.name
