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

# The moved names whose signature changed too (they take the neutral Usage).
TAKES_USAGE = {("app.llm.prices", "cost"), ("app.llm.prices", "token_details")}
BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("old", "name"), sorted(MOVED), ids=[f"{m}.{n}" for m, n in sorted(MOVED)])
def test_old_import_paths_still_work(old: str, name: str) -> None:
    module = importlib.import_module(old)
    new = importlib.import_module(MOVED[(old, name)])
    message = rf"^{old}\.{name} moved to {MOVED[(old, name)]}\.{name}: import it from there"
    if (old, name) in TAKES_USAGE:
        message += (
            r" \(the old name goes in 0\.8\.0\) and now takes app\.llm\.core\.usage\.Usage: "
            r"convert LangChain usage with app\.llm\.langchain\.usage\.from_langchain "
            r"\(pass None, not \{\}, for no usage\)$"
        )
    else:
        message += r" \(the old name goes in 0\.8\.0\)$"
    with pytest.warns(DeprecationWarning, match=message) as caught:
        value = getattr(module, name)
    assert value is getattr(new, name)
    # stacklevel=2: the warning names the line that asked for the old name, not the shim.
    assert [w.filename for w in caught] == [__file__]


def test_a_from_import_of_an_old_name_warns_too() -> None:
    from app.llm.core.prices import cost as new

    with pytest.warns(DeprecationWarning, match="app.llm.prices.cost moved") as caught:
        from app.llm.prices import cost
    assert cost is new
    assert [w.filename for w in caught] == [__file__]


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
    for old in ("app.llm.prices", "app.llm.gateway", "app.llm.errors"):
        with pytest.raises(AttributeError, match="has no attribute 'nope'"):
            _ = importlib.import_module(old).nope


def old_path_uses(path: Path) -> list[str]:
    """`from app.llm.prices import cost`, `prices.cost` where `prices` is the old module,
    `app.llm.prices.cost` after `import app.llm.prices`, and `getattr(prices, "cost")`. A name
    built at run time (`getattr(prices, name)`) is beyond a static check; the warning catches it."""
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
    olds = {old for old, _ in MOVED}

    def module_of(node: ast.expr) -> str | None:
        """The old module an expression names: a bound local (`prices`) or a dotted path
        (`app.llm.prices`, after `import app.llm.prices`)."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        dotted = ".".join([node.id, *reversed(parts)])
        return bound.get(dotted) or (dotted if dotted in olds else None)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (old := module_of(node.value)):
            if (old, node.attr) in MOVED:
                hits.append(f"{ast.unparse(node.value)}.{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and (old := module_of(node.args[0]))
            and (old, node.args[1].value) in MOVED
        ):
            hits.append(f"getattr({ast.unparse(node.args[0])}, {node.args[1].value!r})")
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
        "import app.llm.prices\n"
        "from app.llm import prices\n"
        "from app.llm.gateway import key_setting\n"
        "prices.cost\n"
        "app.llm.prices.ceiling\n"
        "getattr(prices, 'token_details')\n"
        "getattr(app.llm.prices, 'cost')\n"
        "prices.PRICES\n"
    )
    assert sorted(old_path_uses(sample)) == sorted(
        [
            "from app.llm.gateway import key_setting",
            "prices.cost",
            "app.llm.prices.ceiling",
            "getattr(prices, 'token_details')",
            "getattr(app.llm.prices, 'cost')",
        ]
    )


def test_the_old_errors_module_imports_without_a_warning() -> None:
    # Only asking for a moved name warns, not importing the module. LangChain is imported first:
    # on Python 3.14 a dependency of it warns on import, which is not this module's to fix. So an
    # import-time warning in core/errors.py or langchain/errors.py is not caught here; it is
    # accepted (pytest runs with warnings as errors, so the other tests importing them catch it).
    code = (
        "import warnings, app.llm.core.errors, app.llm.langchain.errors\n"
        "warnings.simplefilter('error')\n"
        "import app.llm.errors\n"
        "assert 'install_model_error_handlers' in app.llm.errors.__all__\n"
    )
    subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], check=True
    )
