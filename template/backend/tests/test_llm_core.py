"""The framework-neutral core (#33): the Usage type, the price arithmetic on it, the registry, and
that none of it imports a model framework. Money values are worked out by hand in the comments
(Sonnet shape: input 2, output 10, cache read 0.1, write 5m 2.5, write 1h 4 micro-USD per token).
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from app.llm.core import registry
from app.llm.core.prices import MICRO_USD, Price, Rates, cost, token_details, usd_rates
from app.llm.core.usage import Usage

SONNET = Price(
    Rates(2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000),
    checked="2026-11-06",
    source="https://example.com/pricing",
)
CORE = Path(__file__).resolve().parents[1] / "src" / "app" / "llm" / "core"
# Top-level packages the core must never import (a model framework or a provider SDK).
FRAMEWORKS = frozenset(
    {
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langgraph",
        "langgraph_sdk",
        "langchain_protocol",
        "langsmith",
        "openai",
        "anthropic",
        "agent_framework",
    }
)
# The modules that hold only plain Python: the standard library and each other.
PURE = ("usage", "prices", "registry")


def test_usage_details_default_to_zero_and_it_is_frozen_and_keyword_only() -> None:
    usage = Usage(input_tokens=10, output_tokens=5)
    assert (usage.reasoning_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usage.input_tokens = 11  # type: ignore[misc]
    with pytest.raises(TypeError):
        Usage(10, 5)  # type: ignore[call-arg]


def test_cache_write_tokens_sum_every_lifetime() -> None:
    usage = Usage(
        input_tokens=100,
        output_tokens=0,
        cache_write_short_tokens=20,
        cache_write_long_tokens=30,
        cache_write_undivided_tokens=7,
    )
    assert usage.cache_write_tokens == 57  # 20 + 30 + 7


def test_cost_of_a_usage_with_every_line() -> None:
    # 1,001 input: 601 read, 100 written 5m, 50 written 1h, 50 undivided; plain = 200.
    # 200 x 2 = 400; 601 x 0.1 = 60.1; 100 x 2.5 = 250; (50 + 50) x 4 = 400; 200 out x 10 = 2,000.
    # Total 3,110.1 -> rounded up once: 3,111.
    usage = Usage(
        input_tokens=1_001,
        output_tokens=200,
        reasoning_tokens=80,
        cache_read_tokens=601,
        cache_write_short_tokens=100,
        cache_write_long_tokens=50,
        cache_write_undivided_tokens=50,
    )
    assert cost(SONNET, usage) == 3_111


def test_token_details_of_none_are_none_and_of_a_usage_are_its_counts() -> None:
    assert token_details(None) == {
        "reasoning_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=3,
        cache_read_tokens=4,
        cache_write_long_tokens=2,
    )
    assert token_details(usage) == {
        "reasoning_tokens": 3,
        "cache_read_tokens": 4,
        "cache_write_tokens": 2,
    }


def test_usd_rates_are_whole_micro_usd() -> None:
    # $0.125 per million tokens = 125,000 micro-USD per million.
    assert usd_rates(2, 10, 0.10, 0.125, 4) == Rates(
        2_000_000, 10_000_000, 100_000, 125_000, 4_000_000
    )
    assert MICRO_USD == 1_000_000


def test_key_setting_names_each_provider_and_refuses_others() -> None:
    assert registry.key_setting("openai:any") == "openai_api_key"
    assert registry.key_setting("anthropic:any") == "anthropic_api_key"
    with pytest.raises(
        ValueError,
        match=r"LLM_MODEL 'mistral:x' names no supported provider \(\['anthropic', 'openai'\]\)$",
    ):
        registry.key_setting("mistral:x")


def test_a_model_without_a_price_is_refused_with_the_table_it_was_looked_up_in() -> None:
    table = {"anthropic:m": SONNET}
    assert registry.chat_price("anthropic:m", table) is SONNET
    assert registry.embedding_price("anthropic:m", table) is SONNET
    with pytest.raises(ValueError, match=r"^LLM_MODEL 'x:y' has no price in app/llm/prices\.py$"):
        registry.chat_price("x:y", table)
    with pytest.raises(
        ValueError,
        match=r"^EMBEDDING_MODEL 'x:y' has no price in app/llm/prices\.py EMBEDDING_PRICES$",
    ):
        registry.embedding_price("x:y", table)


def imported_roots(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


# The one import of a provider SDK the core is allowed (#33 T4), mirroring the import-linter ignore
# in llm-contracts.toml: OpenAI's embeddings call, whose usage LangChain's classes discard.
ALLOWED = frozenset({("openai_embeddings.py", "openai")})


def test_core_imports_no_framework() -> None:
    modules = sorted(CORE.glob("*.py"))
    assert modules, CORE
    for path in modules:
        bad = {
            name
            for name in imported_roots(path)
            if name.split(".")[0] in FRAMEWORKS and (path.name, name) not in ALLOWED
        }
        assert bad == set(), path.name


def test_the_price_arithmetic_is_plain_python() -> None:
    for name in PURE:
        for module in imported_roots(CORE / f"{name}.py"):
            root = module.split(".")[0]
            assert root in sys.stdlib_module_names or module.startswith("app.llm.core"), (
                name,
                module,
            )


def test_importing_the_core_loads_no_framework() -> None:
    # A fresh interpreter: the test process has LangChain loaded already.
    code = (
        "import sys, app.llm.core.prices, app.llm.core.registry, app.llm.core.usage\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FRAMEWORKS)!r}))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
