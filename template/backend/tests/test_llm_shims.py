"""The old import paths (#33): names that moved to app.llm.core still answer at their old path
until 0.8.0 (#35), with a DeprecationWarning that points at the importing line, and give the very
same object. The template's own code never goes through them."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_llm_surface import MOVED

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("old", "name"), sorted(MOVED), ids=[f"{m}.{n}" for m, n in sorted(MOVED)])
def test_old_import_paths_still_work(old: str, name: str) -> None:
    module = importlib.import_module(old)
    new = importlib.import_module(MOVED[(old, name)])
    message = rf"^{old}\.{name} moved to {MOVED[(old, name)]}\.{name}: import it from there"
    with pytest.warns(DeprecationWarning, match=message) as caught:
        value = getattr(module, name)
    assert value is getattr(new, name)
    # stacklevel=2: the warning names the line that asked for the old name, not the shim.
    assert [w.filename for w in caught] == [__file__]


def test_a_from_import_of_an_old_name_warns_too() -> None:
    from app.llm.core.prices import cost as new

    with pytest.warns(DeprecationWarning, match="app.llm.prices.cost moved"):
        from app.llm.prices import cost
    assert cost is new


def test_the_price_table_module_imports_without_a_warning() -> None:
    # A fresh interpreter, every warning an error: importing the table module, and the names a
    # price table is written with, must not warn (only the moved arithmetic does).
    code = (
        "from app.llm.prices import EMBEDDING_PRICES, MICRO_USD, PRICES, Price, Rates, _rates\n"
        "from app.llm.core import prices as core\n"
        "assert (MICRO_USD, Price, Rates, _rates) == "
        "(core.MICRO_USD, core.Price, core.Rates, core.usd_rates)\n"
        "assert PRICES and EMBEDDING_PRICES\n"
    )
    subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-W", "error", "-c", code], check=True
    )


def test_an_unknown_name_is_still_an_attribute_error() -> None:
    for old in ("app.llm.prices", "app.llm.gateway"):
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            _ = importlib.import_module(old).nope


def old_path_uses(path: Path) -> list[str]:
    """`from app.llm.prices import cost`, or `prices.cost` where `prices` is the old module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound: dict[str, str] = {}  # local name -> old module
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                if any(old == full for old, _ in MOVED):
                    bound[alias.asname or alias.name] = full
                if (node.module, alias.name) in MOVED:
                    hits.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and any(old == alias.name for old, _ in MOVED):
                    bound[alias.asname] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            old = bound.get(node.value.id)
            if old and (old, node.attr) in MOVED:
                hits.append(f"{node.value.id}.{node.attr}")
    return hits


def test_the_template_never_uses_an_old_path() -> None:
    # One such use in a module the app or conftest imports would stop the whole test run (the
    # warning is an error at import time), so the template must not have any.
    found = {
        path.relative_to(BACKEND).as_posix(): old_path_uses(path)
        for path in [*BACKEND.glob("src/**/*.py"), *BACKEND.glob("tests/**/*.py")]
        if path.name != "test_llm_shims.py"
    }
    assert {path: uses for path, uses in found.items() if uses} == {}


def test_the_old_path_check_finds_a_use(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from app.llm import prices\nfrom app.llm.prices import ceiling\nprices.cost\n"
    )
    assert old_path_uses(sample) == ["from app.llm.prices import ceiling", "prices.cost"]
