"""The rendered layer: files, answers and the checks it plugs into the base Makefile."""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from conftest import BASE_ANSWERS, TEMPLATE

CACHES = (".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".import_linter_cache",
          "__pycache__")


def test_layer_renders_the_backend_and_its_answers(render):
    project = render()
    for path in ("backend/pyproject.toml", "backend/uv.lock", "backend/src/app/main.py",
                 "backend/tests/conftest.py", "backend/.env.example", "mk/fastapi.mk",
                 ".claude/rules/fastapi.md", ".github/template-owned/fastapi.txt"):
        assert (project / path).is_file(), path
    assert (project / ".python-version").read_text() == "3.14\n"
    answers = yaml.safe_load((project / ".copier-answers.fastapi.yml").read_text())
    assert answers["python_version"] == "3.14"
    assert answers["_src_path"]


def test_python_version_answer_sets_the_toolchain(render):
    assert (render(python_version="3.12") / ".python-version").read_text() == "3.12\n"


def test_no_local_cache_or_venv_is_ever_copied(render):
    leaked = [p for p in render().rglob("*") if p.name in CACHES]
    assert leaked == []


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_layer_refuses_a_project_without_base_answers(render, project):
    with pytest.raises(ValueError, match="Apply house-template-base first"):
        render(project(base=None))


def test_layer_refuses_base_answers_without_the_fastapi_stack(render, project):
    with pytest.raises(ValueError, match="don't list the fastapi stack"):
        render(project(stacks=["nextjs"]))


def test_backend_files_need_no_per_project_values():
    # pyproject.toml and uv.lock stay plain files so Dependabot can update the template directly.
    backend = TEMPLATE / "backend"
    assert not list(backend.rglob("*.jinja"))
    assert 'name = "backend"' in (backend / "pyproject.toml").read_text()


def test_lockfile_is_project_owned_after_the_first_copy():
    config = yaml.safe_load((TEMPLATE.parent / "copier.yml").read_text())
    assert "backend/uv.lock" in config["_skip_if_exists"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_lockfile_matches_pyproject():
    result = subprocess.run(["uv", "lock", "--check", "--directory", str(TEMPLATE / "backend")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mk_file_adds_its_targets_to_the_base_aggregates(render):
    mk = (render() / "mk" / "fastapi.mk").read_text()
    assert "INSTALLS += install-fastapi" in mk
    assert "CHECKS += check-fastapi" in mk
    assert "FAST_CHECKS += check-fast-fastapi" in mk
    # Every target the base Makefile's `make help` shows has a description.
    targets = re.findall(r"^([a-z][a-z0-9-]*):", mk, flags=re.MULTILINE)
    described = re.findall(r"^([a-z][a-z0-9-]*):.*## ", mk, flags=re.MULTILINE)
    assert targets and targets == described
    # Every backend tool runs from the lockfile, so a stale lock fails instead of re-resolving.
    for line in mk.splitlines():
        if line.startswith("\t") and " run " in line:
            assert "$(BACKEND)" in line or "--locked" in line, line


def test_every_template_owned_entry_exists(render):
    project = render()
    lines = (project / ".github/template-owned/fastapi.txt").read_text().splitlines()
    patterns = [ln for ln in lines if ln and not ln.startswith("#")]
    files = [str(p.relative_to(project)) for p in project.rglob("*") if p.is_file()]
    for pattern in patterns:
        assert any(fnmatch.fnmatch(f, pattern) for f in files), pattern


def test_rules_file_loads_only_for_backend_paths(render):
    text = (render() / ".claude/rules/fastapi.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front == {"paths": ["backend/**"]}


def test_env_example_lists_every_setting():
    settings = (TEMPLATE / "backend/src/app/settings.py").read_text()
    fields = re.findall(r"^    ([a-z_]+): ", settings, flags=re.MULTILINE)
    example = (TEMPLATE / "backend/.env.example").read_text().upper()
    for field in fields:
        assert field.upper() in example, field


def test_base_answers_fixture_matches_the_pinned_base_version():
    workflow = (TEMPLATE.parent / ".github/workflows/test.yml").read_text()
    assert f"BASE_VERSION: {BASE_ANSWERS['_commit']}" in workflow


def test_rendered_project_has_no_leftover_template_syntax(render):
    for path in Path(render()).rglob("*"):
        if path.is_file() and path.suffix in {".py", ".toml", ".md", ".mk", ".yml", ".txt"}:
            assert "{{" not in path.read_text(), path
