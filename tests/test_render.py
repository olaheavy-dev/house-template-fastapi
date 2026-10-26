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
    # The one exception: the Dockerfile's Python image follows the python_version answer. Each
    # generated project's Dependabot (base watches every Dockerfile) keeps its digests current.
    assert [p.relative_to(backend) for p in backend.rglob("*.jinja")] == [Path("Dockerfile.jinja")]
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
            # GitHub Actions expressions (${{ ... }}) are meant to be there.
            assert not re.search(r"(?<!\$)\{\{", path.read_text()), path


def test_local_database_is_pinned_and_keeps_its_data():
    compose = yaml.safe_load((TEMPLATE / "backend/compose.yaml").read_text())
    db = compose["services"]["db"]
    assert re.fullmatch(r"pgvector/pgvector:[\w.-]+-pg18[\w-]*@sha256:[0-9a-f]{64}", db["image"])
    # PostgreSQL 18 images keep data under /var/lib/postgresql/18/docker; mounting .../data loses it.
    assert db["volumes"] == ["db-data:/var/lib/postgresql"]
    # Only reachable from this machine.
    assert all(port.startswith("127.0.0.1:") for port in db["ports"])
    # Over TCP: a socket check passes during the image's temporary init-time server, so
    # `docker compose up --wait` could return before the database accepts connections.
    assert "pg_isready -h 127.0.0.1" in db["healthcheck"]["test"][1]


def test_local_storage_is_pinned_private_and_healthchecked():
    compose = yaml.safe_load((TEMPLATE / "backend/compose.yaml").read_text())
    storage = compose["services"]["storage"]
    assert re.fullmatch(r"rustfs/rustfs:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", storage["image"])
    assert all(port.startswith("127.0.0.1:") for port in storage["ports"])
    assert storage["volumes"] == ["storage-data:/data"]
    assert "/health" in storage["healthcheck"]["test"][-1]


def test_migrations_need_no_alembic_ini():
    backend = TEMPLATE / "backend"
    assert not (backend / "alembic.ini").exists()
    assert 'script_location = "migrations"' in (backend / "pyproject.toml").read_text()


@pytest.mark.parametrize("python", ["3.14", "3.13", "3.12"])
def test_the_image_is_pinned_to_the_answered_python_and_runs_as_non_root(render, python):
    dockerfile = (render(python_version=python) / "backend" / "Dockerfile").read_text()
    images = re.findall(r"^FROM (\S+)", dockerfile, flags=re.MULTILINE)
    # uv, then the build and final stages on the same Python, everything pinned by digest.
    assert len(images) == 3
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in images)
    assert images[1] == images[2]
    assert images[1].startswith(f"python:{python}-slim-trixie@")
    assert re.search(r"^USER 10001:10001$", dockerfile, flags=re.MULTILINE)
    # Built from the lockfile, installed as a package (no source tree needed at runtime).
    assert dockerfile.count("uv sync --locked") == 2
    assert "--no-editable" in dockerfile


def test_secrets_and_tests_stay_out_of_the_image(render):
    ignored = (render() / "backend" / ".dockerignore").read_text().splitlines()
    for entry in (".env", ".env.*", ".venv/", "tests/"):
        assert entry in ignored


def test_the_api_waits_for_migrations_that_run_once(render):
    compose = yaml.safe_load((render() / "backend" / "compose.yaml").read_text())
    migrate, api = compose["services"]["migrate"], compose["services"]["api"]
    assert migrate["command"] == ["alembic", "upgrade", "heads"]
    assert migrate["restart"] == "no"
    assert api["depends_on"]["migrate"] == {"condition": "service_completed_successfully"}
    assert all(port.startswith("127.0.0.1:") for port in api["ports"])


@pytest.mark.parametrize("images", [True, False])
def test_the_publish_workflow_follows_the_base_answer(render, project, images):
    dst = render(project(container_images=images))
    workflow = dst / ".github" / "workflows" / "publish-api.yml"
    owned = (dst / ".github/template-owned/fastapi.txt").read_text()
    assert workflow.exists() is images
    assert (".github/workflows/publish-api.yml" in owned) is images
    if images:
        jobs = yaml.safe_load(workflow.read_text())["jobs"]
        assert jobs["api"]["uses"] == "./.github/workflows/publish-image.yml"
        assert jobs["api"]["with"] == {"image": "api", "context": "backend"}


def test_a_local_env_file_never_reaches_a_project(render):
    # template/backend/.env holds a developer's keys for recording and evals. A real one serves as
    # the probe; otherwise a throwaway one is written for the test.
    secret = TEMPLATE / "backend" / ".env"
    created = not secret.exists()
    if created:
        secret.write_text("OPENAI_API_KEY=sk-never-copied\n")
    try:
        project = render()
    finally:
        if created:
            secret.unlink()
    assert not (project / "backend" / ".env").exists()
    assert (project / "backend" / ".env.example").exists()
