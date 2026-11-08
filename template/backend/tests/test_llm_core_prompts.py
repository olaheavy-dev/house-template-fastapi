"""The neutral prompt (#33 T5a): its version hash, tag and metadata, and loading the file.
Nothing here needs a model framework; the LangChain subclass is tested in test_llm_surface.py."""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

from app.auth import User
from app.llm.core import prompts as core
from app.llm.gateway import Gateway
from app.llm.langchain.prompts import Prompt as LangChainPrompt
from app.llm.langchain.prompts import load_prompt
from app.settings import Settings
from tests.test_llm_core import imported_roots
from tests.test_llm_core_embeddings import loaded_by

CORE_PROMPTS = Path(core.__file__)


def test_the_version_is_the_first_12_hex_of_the_sha256_of_the_text() -> None:
    # printf %s 'Say hello.' | shasum -a 256 -> c8e2c1437abb...
    assert core.Prompt("greet", "Say hello.").sha == "c8e2c1437abb"
    # printf %s '' | shasum -a 256 -> e3b0c44298fc... (the empty text still has a version)
    assert core.Prompt("empty", "").sha == "e3b0c44298fc"
    # printf 'Answer.\n' | shasum -a 256 -> 8b50097a1438...
    assert core.Prompt("ask", "Answer.\n").sha == "8b50097a1438"
    # printf %s 'Answer briefly.' | shasum -a 256 -> e68562472088...
    assert core.Prompt("ask", "Answer briefly.").sha == "e68562472088"


def test_the_tag_and_the_metadata_carry_the_name_and_the_version() -> None:
    prompt = core.Prompt("greet", "Say hello.")
    assert prompt.tag == "prompt:greet@c8e2c1437abb"
    assert prompt.metadata() == {"prompt": "greet", "prompt_sha": "c8e2c1437abb"}


def test_a_prompt_cannot_be_changed() -> None:
    prompt = core.Prompt("greet", "Say hello.")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.text = "Say goodbye."  # type: ignore[misc]


def test_prompt_text_reads_the_file_unstripped(tmp_path: Path) -> None:
    (tmp_path / "ask.md").write_bytes("Café ☕\n\n".encode())
    assert core.prompt_text("ask", tmp_path) == "Café ☕\n\n"


def test_prompt_text_refuses_a_nested_or_escaping_name_and_a_missing_file(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.md").write_text("x")
    (tmp_path.parent / "x.md").write_text("outside")
    for name in ("sub/x", "../x"):
        with pytest.raises(ValueError, match="not a prompt name"):
            core.prompt_text(name, tmp_path)
    with pytest.raises(FileNotFoundError):
        core.prompt_text("absent", tmp_path)


def test_the_prompts_directory_is_app_prompts() -> None:
    assert Path(__file__).resolve().parents[1] / "src" / "app" / "prompts" == core.PROMPTS.resolve()


def test_load_prompt_gives_the_langchain_prompt_which_is_a_core_prompt(tmp_path: Path) -> None:
    (tmp_path / "ask.md").write_text("Answer.\n", encoding="utf-8")
    loaded = load_prompt("ask", tmp_path)
    assert isinstance(loaded, LangChainPrompt)
    assert isinstance(loaded, core.Prompt)
    assert loaded.sha == "8b50097a1438"


def test_call_config_accepts_a_plain_core_prompt() -> None:
    gateway = Gateway(Settings(_env_file=None, app_env="test"), lambda: None)  # type: ignore[arg-type,return-value]
    config = gateway.call_config(User(id="u_9"), "answer", core.Prompt("ask", "Answer briefly."))
    assert config["run_name"] == "ask"
    assert config["tags"] == ["prompt:ask@e68562472088"]
    assert config["metadata"] == {
        "prompt": "ask",
        "prompt_sha": "e68562472088",
        "feature": "answer",
        "user_ref": "u_9",
    }


def test_the_neutral_prompt_imports_only_the_standard_library() -> None:
    for module in imported_roots(CORE_PROMPTS):
        assert module.split(".")[0] in sys.stdlib_module_names, module
    tree = ast.parse(CORE_PROMPTS.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level]


def test_importing_the_neutral_prompt_loads_no_framework() -> None:
    assert loaded_by("app.llm.core.prompts") == "[]"
