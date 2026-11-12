"""The shared parity tests (backend/tests/test_llm_parity_*.py, #35) must stay shared: they state
what every model framework does, so they import no framework and never skip. Every other
test_llm_*.py file is classified as one framework's own, so a new file has to be placed on purpose.
"""

from __future__ import annotations

import ast
import re

import pytest
from conftest import TEMPLATE

TESTS = TEMPLATE / "backend" / "tests"
PARITY = sorted(TESTS.glob("test_llm_parity_*.py"))
# The harness protocol module and the harness files of each framework. Parity files may import the
# protocol (`tests.harness`) and nothing else under tests.harness.
HARNESS = sorted((TESTS / "harness").glob("*.py"))
PROTOCOL = TESTS / "harness" / "__init__.py"
FRAMEWORK_SEGMENT = re.compile(r"^(langchain\w*|langgraph\w*|langsmith|agent_framework\w*)$")
FRAMEWORK_PREFIXES = ("app.llm.langchain", "app.llm.maf")
HARNESS_MODULES = {"langchain", "maf"} | {p.stem for p in HARNESS if p.stem != "__init__"}
SKIPS = {"skip", "skipif", "xfail", "importorskip"}


def framework_import_problems(source: str) -> list[str]:
    """What in `source` imports a framework, or could hide one: any dotted segment naming a
    framework, a framework adapter or a harness module other than the protocol, a relative import,
    and a dynamic import."""
    problems = []
    for node in ast.walk(ast.parse(source)):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"relative import at line {node.lineno}")
                continue
            modules = [node.module or ""]
            # `from tests import harness` and `from app.llm import langchain` name a module too.
            modules += [f"{node.module}.{alias.name}" for alias in node.names]
            if node.module == "tests.harness":
                # Names the protocol exports are fine; a module of a framework is not.
                modules = [
                    m
                    for m in modules
                    if m.rsplit(".", 1)[-1] in HARNESS_MODULES or m == node.module
                ]
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"import_module", "__import__"}:
                problems.append(f"dynamic import at line {node.lineno}")
        for module in modules:
            if any(FRAMEWORK_SEGMENT.match(part) for part in module.split(".")):
                problems.append(f"framework import {module}")
            elif module.startswith(FRAMEWORK_PREFIXES):
                problems.append(f"framework adapter import {module}")
            elif module.startswith("tests.harness.") and module != "tests.harness":
                problems.append(f"harness module import {module}")
    return problems


def skip_problems(source: str) -> list[str]:
    """Where `source` skips or expects failure (exact names, bare or as attributes), or branches
    on which framework runs (`framework.name`)."""
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            if node.attr in SKIPS:
                problems.append(f"{node.attr} at line {node.lineno}")
            elif node.attr == "name" and getattr(node.value, "id", "") == "framework":
                problems.append(f"framework.name at line {node.lineno}")
        elif isinstance(node, ast.Name) and node.id in SKIPS:
            problems.append(f"{node.id} at line {node.lineno}")
        elif isinstance(node, ast.alias) and node.name in SKIPS:
            problems.append(f"{node.name} imported")
    return problems


def test_parity_files_import_no_framework():
    assert PARITY, "no parity files found"
    for path in PARITY:
        assert not framework_import_problems(path.read_text()), path.name
    # The protocol module loads a framework's harness by name (a dynamic import, on purpose), so
    # only its static imports are held to the rule.
    protocol = [p for p in framework_import_problems(PROTOCOL.read_text()) if "dynamic" not in p]
    assert not protocol, protocol


def test_parity_and_harness_files_never_skip():
    assert HARNESS, "no harness files found"
    for path in [*PARITY, *HARNESS]:
        assert not skip_problems(path.read_text()), path.name


@pytest.mark.parametrize(
    "source",
    [
        "import langchain_core",
        "import langchain_core.messages",
        "from langchain_core.messages import AIMessage",
        "from app.llm.langchain import gateway",
        "from app.llm import langchain",
        "from app.llm.langchain.gateway import Gateway",
        "from app.llm.maf import client",
        "import agent_framework",
        "from agent_framework.openai import OpenAIChatClient",
        "from app.thing.langgraph import graph",
        "import langsmith.client",
        "from tests.harness.langchain import HARNESS",
        "from tests.harness import langchain",
        "import tests.harness.maf",
        "from . import something",
        "from .harness import Harness",
        "import importlib\nimportlib.import_module('langchain_core')",
        "from importlib import import_module\nimport_module('x')",
        "__import__('agent_framework')",
    ],
)
def test_the_import_guard_catches(source):
    assert framework_import_problems(source), source


@pytest.mark.parametrize(
    "source",
    [
        "from tests.harness import Harness, Reply, user",
        "import tests.harness",
        "from app.llm.core import budget",
        "from app.llm import prices",
        "import openai",
    ],
)
def test_the_import_guard_allows(source):
    assert not framework_import_problems(source), source


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n@pytest.mark.skip\ndef test_a(): ...",
        "import pytest\n@pytest.mark.skipif(True, reason='x')\ndef test_a(): ...",
        "import pytest\n@pytest.mark.xfail\ndef test_a(): ...",
        "import pytest\npytest.importorskip('x')",
        "import pytest\npytest.skip('x')",
        "from pytest import skip\nskip('x')",
        "from pytest import importorskip",
        "pytestmark = skipif(True)",
        "def test_a(framework):\n    if framework.name == 'maf': ...",
    ],
)
def test_the_skip_guard_catches(source):
    assert skip_problems(source), source


def test_the_skip_guard_allows_a_name_that_only_contains_skip():
    source = "skipped_calls = 0\ndef was_skipped(): return skipped_calls\nx = obj.skipped_calls"
    assert not skip_problems(source)


# The files as of #35 PR 2. A new test_llm_*.py is a parity test (name it test_llm_parity_*.py), or
# is added to CORE (framework-free: it runs whichever framework the project chose, and the test
# below checks it imports none) or to LANGCHAIN_ONLY / MAF_ONLY (one framework's own adapter).
CORE = {
    "test_llm_core_budget.py",
    "test_llm_core_embeddings.py",
    "test_llm_cost_pins.py",
    "test_llm_core_errors.py",
    "test_llm_prices.py",
    "test_llm_core_redaction.py",
    "test_llm_core_redaction_definitions.py",
    "test_llm_core_steps.py",
    "test_llm_core.py",
    "test_llm_errors.py",
    "test_llm_redaction_corpus.py",
    "test_llm_report_cli.py",
    "test_llm_settings.py",
}
LANGCHAIN_ONLY = {
    "test_llm_agents.py",
    "test_llm_budget.py",
    "test_llm_checkpoints.py",
    "test_llm_core_prompts.py",
    "test_llm_embeddings.py",
    "test_llm_error_logs.py",
    "test_llm_fakes.py",
    "test_llm_gateway.py",
    "test_llm_langchain_callbacks.py",
    "test_llm_langchain_embeddings.py",
    "test_llm_langchain_errors.py",
    "test_llm_ledger.py",
    "test_llm_migrations.py",
    "test_llm_redaction_agents.py",
    "test_llm_redaction_context.py",
    "test_llm_redaction_embeddings.py",
    "test_llm_redaction_fields.py",
    "test_llm_redaction_ledger.py",
    "test_llm_redaction_logs.py",
    "test_llm_redaction_model.py",
    "test_llm_redaction_observability.py",
    "test_llm_redaction_restore.py",
    "test_llm_redaction_stream.py",
    "test_llm_redaction_structured.py",
    "test_llm_redaction_tools.py",
    "test_llm_redaction_traces.py",
    "test_llm_redaction_wire.py",
    "test_llm_steps.py",
    "test_llm_surface.py",
    "test_llm_usage.py",
}
MAF_ONLY = {
    "test_llm_maf_cassettes.py",
    "test_llm_maf_errors.py",
    "test_llm_maf_gateway.py",
    "test_llm_maf_options.py",
    "test_llm_maf_redaction.py",
    "test_llm_maf_stream.py",
    "test_llm_maf_surface.py",
    "test_llm_maf_usage.py",
    "test_llm_maf_wire.py",
}


def test_core_files_import_no_framework():
    for name in CORE:
        problems = [
            p for p in framework_import_problems((TESTS / name).read_text()) if "dynamic" not in p
        ]
        assert not problems, (name, problems)


def test_every_llm_test_file_is_classified():
    found = {p.name for p in TESTS.glob("test_llm_*.py")}
    classified = {p.name for p in PARITY} | CORE | LANGCHAIN_ONLY | MAF_ONLY
    assert found <= classified, f"classify: {sorted(found - classified)}"
    assert classified - {p.name for p in PARITY} <= found, "stale entry"
    assert not CORE & (LANGCHAIN_ONLY | MAF_ONLY), "a file is in two sets"


def test_copier_excludes_exactly_the_langchain_only_tests_from_other_frameworks():
    # copier.yml lists LangChain's own test files one by one (`llm != 'langchain'`), and MAF's by
    # pattern: the lists must agree with these sets, or a test file would reach a project whose
    # framework it cannot import.
    import yaml

    from conftest import ROOT

    excluded = yaml.safe_load((ROOT / "copier.yml").read_text())["_exclude"]
    pattern = re.compile(
        r"\{% if llm is defined and llm != 'langchain' %\}backend/tests/(test_llm_\w+\.py)\{% endif %\}"
    )
    listed = {m.group(1) for entry in excluded if (m := pattern.fullmatch(entry))}
    # test_llm_redaction_observability.py and test_llm_redaction_wire.py are in the same list.
    assert listed == LANGCHAIN_ONLY
    assert "{% if llm is defined and llm != 'maf' %}backend/tests/test_llm_maf_*.py{% endif %}" in excluded
    assert {p.name for p in TESTS.glob("test_llm_maf_*.py")} == MAF_ONLY
