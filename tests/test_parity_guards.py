"""The shared parity tests (backend/tests/test_llm_parity_*.py, #35) must stay shared: they state
what every model framework does, so they import no framework and never skip. Every other
test_llm_*.py file is classified as one framework's own, so a new file has to be placed on purpose.
"""

from __future__ import annotations

import ast
import re

from conftest import TEMPLATE

TESTS = TEMPLATE / "backend" / "tests"
PARITY = sorted(TESTS.glob("test_llm_parity_*.py"))
# Imported by the parity files, directly or through the harness protocol.
FRAMEWORKS = re.compile(r"^(langchain\w*|langgraph\w*|langsmith|agent_framework\w*)$")


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or ""


def test_parity_files_import_no_framework():
    assert PARITY, "no parity files found"
    for path in [*PARITY, TESTS / "harness" / "__init__.py"]:
        for module in _imports(path):
            assert not FRAMEWORKS.match(module.split(".")[0]), (path.name, module)


def test_parity_files_never_skip():
    for path in PARITY:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
            assert not re.search(r"skip|xfail|importorskip", name or ""), (path.name, node.lineno)


# The files that test one framework's own adapter or the framework-free core, as of #35 PR 1. A
# new test_llm_*.py is a parity test (name it test_llm_parity_*.py) or is added to one of these.
LANGCHAIN_ONLY = {
"test_llm_agents.py",
"test_llm_budget.py",
"test_llm_checkpoints.py",
"test_llm_core_budget.py",
"test_llm_core_embeddings.py",
"test_llm_core_errors.py",
"test_llm_core_prompts.py",
"test_llm_core_redaction.py",
"test_llm_core_steps.py",
"test_llm_core.py",
"test_llm_cost_pins.py",
"test_llm_embeddings.py",
"test_llm_error_logs.py",
"test_llm_errors.py",
"test_llm_fakes.py",
"test_llm_gateway.py",
"test_llm_langchain_callbacks.py",
"test_llm_langchain_embeddings.py",
"test_llm_langchain_errors.py",
"test_llm_ledger.py",
"test_llm_migrations.py",
"test_llm_prices.py",
"test_llm_redaction_agents.py",
"test_llm_redaction_context.py",
"test_llm_redaction_corpus.py",
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
"test_llm_report_cli.py",
"test_llm_settings.py",
"test_llm_steps.py",
"test_llm_surface.py",
"test_llm_usage.py",
}
MAF_ONLY: set[str] = set()


def test_every_llm_test_file_is_classified():
    found = {p.name for p in TESTS.glob("test_llm_*.py")}
    classified = {p.name for p in PARITY} | LANGCHAIN_ONLY | MAF_ONLY
    assert found <= classified, f"classify: {sorted(found - classified)}"
    assert classified - {p.name for p in PARITY} <= found, "stale entry"
