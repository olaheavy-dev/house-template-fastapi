"""The answers that switch parts of the backend on and off (#29): `database` and `storage`.

Docs: Copier 9.19.1
- `_exclude` (each pattern can be templated; defining it replaces Copier's defaults), the `.jinja`
  suffix rule (a file next to one with the suffix is ignored) and `_skip_if_exists:
  https://copier.readthedocs.io/en/v9.19.1/configuring/
- Updating: https://copier.readthedocs.io/en/v9.19.1/updating/
uv extras: https://docs.astral.sh/uv/concepts/projects/sync/
Compose `include`: https://docs.docker.com/reference/compose-file/include/

How a part is switched
- Files that only exist with the part are listed in copier.yml `_exclude` as Jinja patterns.
- Files whose content changes with the answer are `x.jinja`. Where the template's own in-place
  loop (record, eval, test inside template/backend) must run them, a plain all-on twin `x` sits
  next to `x.jinja` (Copier ignores the twin when it renders), and a test keeps the two equal.
- Dependencies are optional extras in the plain pyproject.toml; the image and make targets
  install only the answered extras.
"""

from __future__ import annotations

import ast
import fnmatch
import itertools
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from conftest import BASE_ANSWERS, TEMPLATE

ON_OFF = pytest.mark.parametrize("storage", [True, False], ids=["storage-on", "storage-off"])
# Every combination of the two answers.
COMBINATIONS = list(itertools.product([True, False], [True, False]))
ANSWERS = pytest.mark.parametrize(
    ("database", "storage"), COMBINATIONS,
    ids=[f"database-{'on' if d else 'off'}-storage-{'on' if s else 'off'}" for d, s in COMBINATIONS])

STORAGE_FILES = [
    "backend/src/app/storage.py",
    "backend/tests/storage.py",
    "backend/tests/test_storage.py",
    "backend/compose/storage/compose.yaml",
]
# The paths `database: false` removes, as patterns (fnmatch: `*` also crosses `/`). Each must match
# something that is removed, and everything removed must match one (test_database_files_are_exact).
DATABASE_PATTERNS = [
    "backend/src/app/db.py",
    "backend/src/app/models.py",
    "backend/src/app/llm/*",
    "backend/src/app/prompts/*",
    "backend/migrations/*",
    "backend/compose/db/*",
    "backend/tests/db.py",
    "backend/tests/models.py",
    "backend/tests/recording.py",
    "backend/tests/test_db.py",
    "backend/tests/test_llm_*.py",
    "backend/tests/test_recording_keys.py",
    "backend/tests/test_observability.py",
    "backend/tests/evals/*",
    "backend/tests/cassettes/*",
]
# One file from each pattern that must be absent without the database.
DATABASE_FILES = [
    "backend/src/app/db.py",
    "backend/src/app/models.py",
    "backend/src/app/llm/gateway.py",
    "backend/src/app/prompts/README.md",
    "backend/migrations/env.py",
    "backend/migrations/versions/20261026_app_project_root.py",
    "backend/compose/db/compose.yaml",
    "backend/tests/db.py",
    "backend/tests/models.py",
    "backend/tests/recording.py",
    "backend/tests/test_db.py",
    "backend/tests/test_llm_gateway.py",
    "backend/tests/test_llm_errors.py",
    "backend/tests/test_recording_keys.py",
    "backend/tests/test_observability.py",
    "backend/tests/evals/conftest.py",
    "backend/tests/cassettes/test_llm_gateway/test_a_replayed_call_is_answered_and_its_cost_settled.yaml",
]
# Files that exist in both renders but change with the answer, each with an all-on plain twin.
TWINS = [
    "backend/src/app/main.py",
    "backend/src/app/settings.py",
    "backend/src/app/log.py",
    "backend/tests/conftest.py",
    "backend/.env.example",
]
# Words that must not appear in a storage-off project, except in pyproject.toml and uv.lock
# (the dependency stays declared: `boto3` in the extra, `types-boto3` in the dev group).
STORAGE_WORDS = re.compile(r"boto|STORAGE_|storage_settings|presigned|RustFS", re.IGNORECASE)
NOT_COMPARED = {"backend/pyproject.toml", "backend/uv.lock"}


def files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


@ANSWERS
def test_render_matrix(render, database, storage):
    project = render(database=database, storage=storage)
    for path in DATABASE_FILES:
        assert (project / path).exists() is database, path
    for path in STORAGE_FILES:
        assert (project / path).exists() is storage, path
    # Always there, whatever the answers.
    for path in ("backend/.env.example", "backend/compose.yaml", "backend/src/app/main.py",
                 "backend/tests/conftest.py", "backend/src/app/api/__init__.py", "mk/fastapi.mk"):
        assert (project / path).is_file(), path
    # The suffix is always stripped.
    assert [p for p in files(project) if p.endswith(".jinja")] == []
    # The dependencies are declared in one plain pyproject.toml either way.
    pyproject = read_toml(project / "backend/pyproject.toml")
    assert pyproject["project"]["optional-dependencies"]["storage"][0].startswith("boto3")
    assert not any(d.startswith("boto3") for d in pyproject["project"]["dependencies"])
    mk = (project / "mk/fastapi.mk").read_text()
    assert ("storage-up:" in mk) is storage
    answers = yaml.safe_load((project / ".copier-answers.fastapi.yml").read_text())
    assert (answers["database"], answers["storage"]) == (database, storage)


def render_other(render, tmp_path: Path, **answers) -> Path:
    """A second render in its own directory (the render fixture's own is `project`)."""
    dst = tmp_path / "other"
    dst.mkdir()
    (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump(BASE_ANSWERS))
    return render(dst, **answers)


def test_database_files_are_exact(render, tmp_path):
    # Everything `database: false` removes matches DATABASE_PATTERNS, and every pattern removes
    # something: a new file under a database path, or a pattern that matches nothing, fails here.
    on = set(files(render(database=True, storage=True)))
    off = set(files(render_other(render, tmp_path, database=False, storage=True)))
    removed = on - off
    assert (off - on) == set()
    stray = [p for p in removed if not any(fnmatch.fnmatch(p, pat) for pat in DATABASE_PATTERNS)]
    assert stray == []
    unused = [pat for pat in DATABASE_PATTERNS if not any(fnmatch.fnmatch(p, pat) for p in removed)]
    assert unused == []


def read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def test_storage_is_on_by_default(render):
    assert (render() / "backend/src/app/storage.py").is_file()


def test_database_is_on_by_default(render):
    project = render()
    assert (project / "backend/src/app/db.py").is_file()
    answers = yaml.safe_load((project / ".copier-answers.fastapi.yml").read_text())
    assert answers["database"] is True


def test_jinja_twins_render_to_their_plain_file(render):
    # With storage on, the rendered .jinja file is byte-identical to the plain file the in-place
    # loop runs. Also: every .jinja next to a plain file is listed here (no unguarded twin) and
    # every listed twin really has both files.
    project = render(storage=True)
    pairs = sorted(
        p.relative_to(TEMPLATE).as_posix()
        for p in TEMPLATE.rglob("*.jinja")
        if p.with_suffix("").exists()
    )
    assert pairs == sorted(t + ".jinja" for t in TWINS)
    for twin in TWINS:
        assert (project / twin).read_bytes() == (TEMPLATE / twin).read_bytes(), twin


SETTING = re.compile(r"^    ([a-z][a-z0-9_]*): ", re.MULTILINE)
EXAMPLE = re.compile(r"^#? ?([A-Z][A-Z0-9_]*)=", re.MULTILINE)
# Variables .env.example documents that are not Settings fields: read by Compose, the tests or
# the OpenTelemetry SDK.
NOT_SETTINGS = {"DB_PORT", "TEST_DATABASE_URL", "FASTAPI_OTEL_AUTO_CONFIGURE", "OTEL_SERVICE_NAME",
                "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"}
STORAGE_NOT_SETTINGS = {"STORAGE_PORT", "STORAGE_CONSOLE_PORT"}


@ON_OFF
def test_env_example_matches_settings(render, storage):
    project = render(storage=storage)
    fields = {f.upper() for f in SETTING.findall((project / "backend/src/app/settings.py").read_text())}
    listed = set(EXAMPLE.findall((project / "backend/.env.example").read_text()))
    allowed = NOT_SETTINGS | (STORAGE_NOT_SETTINGS if storage else set())
    assert fields - listed == set(), "settings missing from .env.example"
    assert listed - fields - allowed == set(), "variables in .env.example that are not settings"
    assert any(f.startswith("STORAGE_") for f in fields) is storage


# Template-owned files that exist only with the database (the project's own tests are not listed).
DATABASE_OWNED = [
    "backend/src/app/llm/__init__.py",
    "backend/src/app/llm/gateway.py",
    "backend/migrations/env.py",
    "backend/migrations/script.py.mako",
    "backend/migrations/versions/20261026_app_project_root.py",
    "backend/compose/db/compose.yaml",
    "backend/tests/db.py",
    "backend/tests/evals/conftest.py",
]


@ANSWERS
def test_every_template_owned_entry_exists(render, database, storage):
    project = render(database=database, storage=storage)
    owned = (project / ".github/template-owned/fastapi.txt").read_text().splitlines()
    patterns = [ln for ln in owned if ln and not ln.startswith("#")]
    existing = list(files(project))
    for pattern in patterns:
        assert any(fnmatch.fnmatch(f, pattern) for f in existing), pattern
    # And the reverse for the parts that switch: a file is listed exactly when it exists.
    for path in DATABASE_OWNED:
        assert (path in patterns) is database, path
    for path in STORAGE_FILES:
        if path != "backend/tests/test_storage.py":  # the project's own tests are not listed
            assert (path in patterns) is storage, path


@ANSWERS
def test_rendered_project_has_no_leftover_template_syntax(render, database, storage):
    for path in files(render(database=database, storage=storage)).values():
        if path.suffix in {".py", ".toml", ".md", ".mk", ".yml", ".yaml", ".txt", ".example"}:
            # GitHub Actions expressions (${{ ... }}) are meant to be there.
            assert not re.search(r"(?<!\$)\{\{|\{%", path.read_text()), path


@ANSWERS
def test_every_rendered_python_file_parses(render, database, storage):
    project = render(database=database, storage=storage)
    for rel, path in files(project).items():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=rel)


def test_no_part_left_behind_when_storage_is_off(render):
    project = render(storage=False)
    found = []
    for rel, path in files(project).items():
        if rel in NOT_COMPARED or ".git/" in rel:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        found += [(rel, m.group(0)) for m in STORAGE_WORDS.finditer(text)]
    assert found == []
    # The mentions that remain are only the declared dependency.
    assert "types-boto3" in (project / "backend/pyproject.toml").read_text()


def test_no_part_left_behind_check_catches_a_leftover(render):
    # The word list does find what it is meant to find (a check that cannot fail proves nothing).
    project = render(storage=True)
    hits = [rel for rel, p in files(project).items() if rel not in NOT_COMPARED
            and p.suffix in {".py", ".md", ".mk", ".yaml", ".example"}
            and STORAGE_WORDS.search(p.read_text())]
    assert "backend/src/app/storage.py" in hits and "mk/fastapi.mk" in hits
    assert ".claude/rules/fastapi.md" in hits and "backend/compose.yaml" in hits


def extras_flags(database: bool, storage: bool) -> str:
    return " ".join(
        (["--extra db", "--extra langchain"] if database else []) + (["--extra storage"] if storage else []))


@ANSWERS
def test_the_image_installs_the_answered_extras(render, database, storage):
    project = render(database=database, storage=storage)
    flags = extras_flags(database, storage)
    dockerfile = (project / "backend/Dockerfile").read_text()
    syncs = [ln for ln in dockerfile.splitlines() if "uv sync" in ln]
    assert len(syncs) == 2
    for ln in syncs:
        assert ln.endswith(" ".join(["--no-editable", flags]).strip()), ln
    mk = (project / "mk/fastapi.mk").read_text()
    runs = [ln for ln in mk.splitlines() if "uv --directory backend" in ln and " sync " in ln
            or ln.startswith(("BACKEND :=", "LIVE :="))]
    # Every uv call that installs or runs the project goes through EXTRAS, set from the answers.
    # LIVE (recording and evals) exists only with the database.
    assert len(runs) == (3 if database else 2) and all("$(EXTRAS)" in ln for ln in runs), runs
    assert ("LIVE :=" in mk) is database
    extras = re.search(r"^EXTRAS :=(.*)$", mk, flags=re.MULTILINE)
    assert extras and extras.group(1).strip() == flags
    if shutil.which("make"):
        # Make itself expands the line (a missing space once glued `--locked--extra`).
        dry = subprocess.run(["make", "-n", "-f", "mk/fastapi.mk", "install-fastapi", "dev"],
                             cwd=project, capture_output=True, text=True)
        assert dry.returncode == 0, dry.stderr
        want = " ".join(["--locked", flags]).strip()
        lines = [ln for ln in dry.stdout.splitlines() if ln.startswith("uv ")]
        assert len(lines) == 2 and all(want in ln for ln in lines), lines
        assert all(("--extra" in ln) is bool(flags) for ln in lines), lines
    # `make lock` locks every extra whatever the answers (the lockfile is not per answer).
    assert "uv --directory backend lock\n" in mk


ALWAYS_TARGETS = {"install-fastapi", "check-fastapi", "check-fast-fastapi", "lock", "dev", "test",
                  "image", "up"}
DATABASE_TARGETS = {"db-up", "migrate", "migration", "llm-report", "record-llm", "eval"}


@ANSWERS
def test_make_targets_follow_the_answers(render, database, storage):
    mk = (render(database=database, storage=storage) / "mk/fastapi.mk").read_text()
    want = (ALWAYS_TARGETS | (DATABASE_TARGETS if database else set())
            | ({"storage-up"} if storage else set()))
    defined = set(re.findall(r"^([a-z][a-z0-9-]*):", mk, flags=re.MULTILINE))
    phony = set(re.search(r"^\.PHONY: (.*)$", mk, flags=re.MULTILINE).group(1).split())
    assert defined == want
    assert phony == want
    up_help = re.search(r"^up:.*## (.*)$", mk, flags=re.MULTILINE).group(1)
    assert ("database" in up_help) is database
    # mypy also reads migrations/, which exists only with the database (a missing path is an error).
    mypy = [ln.strip() for ln in mk.splitlines() if ln.strip().endswith(" mypy src tests")
            or ln.strip().endswith(" mypy src tests migrations")]
    assert len(mypy) == 2
    assert all(ln.endswith("migrations") is database for ln in mypy), mypy
    if database:
        assert "comma := ," in mk
    else:
        assert not any(word in mk for word in ("comma", "record-llm", "LLM_", "alembic"))


def test_dependencies_live_in_the_answered_extras():
    # The parts that can be switched off are extras; the main dependencies are only what every
    # project needs. The lists below are the ones that moved, not new packages (#29).
    pyproject = read_toml(TEMPLATE / "backend/pyproject.toml")
    def names(requirements):
        return {re.split(r"[\[<>=]", r)[0] for r in requirements}
    assert names(pyproject["project"]["dependencies"]) == {"fastapi", "pydantic-settings", "structlog"}
    extras = pyproject["project"]["optional-dependencies"]
    assert set(extras) == {"db", "langchain", "storage"}
    assert names(extras["db"]) == {"alembic", "sqlalchemy", "psycopg", "psycopg-pool", "pgvector"}
    assert names(extras["langchain"]) == {
        "langgraph-checkpoint-postgres", "langchain", "langchain-core", "langchain-openai",
        "langchain-anthropic", "langgraph", "langsmith"}
    assert names(extras["storage"]) == {"boto3"}
    dev = names(pyproject["dependency-groups"]["dev"])
    assert {"testcontainers", "types-boto3", "langsmith", "vcrpy", "pytest-recording"} <= dev
    # The import contract names the gateway, which a project without the database lacks.
    contract = pyproject["tool"]["importlinter"]["contracts"][0]
    assert contract["unmatched_ignore_imports_alerting"] == "none"


def compose_config(project: Path) -> dict:
    result = subprocess.run(
        ["docker", "compose", "-f", str(project / "backend/compose.yaml"), "config", "--format",
         "json"], capture_output=True, text=True, cwd=project / "backend")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@ANSWERS
def test_compose_services_follow_the_answers(render, database, storage):
    project = render(database=database, storage=storage)
    compose = yaml.safe_load((project / "backend/compose.yaml").read_text())
    includes = [i if isinstance(i, str) else i["path"] for i in compose.get("include", [])]
    assert includes == (["compose/db/compose.yaml"] if database else []) + (
        ["compose/storage/compose.yaml"] if storage else [])
    assert ("include" in compose) is (database or storage)
    api = compose["services"]["api"]
    assert any(k.startswith("STORAGE_") for k in api["environment"]) is storage
    assert api["environment"]["APP_ENV"] == "local"
    assert api["environment"]["LOG_JSON"] == "false"
    assert ("DATABASE_URL" in api["environment"]) is database
    assert ("depends_on" in api) is database
    assert set(compose["services"]) == {"api"} | ({"migrate"} if database else set())
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    config = compose_config(project)
    assert set(config["services"]) == (
        {"api"} | ({"db", "migrate"} if database else set()) | ({"storage"} if storage else set()))
    if database:
        # depends_on still reaches the included database service.
        assert config["services"]["migrate"]["depends_on"]["db"]["condition"] == "service_healthy"
        assert config["services"]["api"]["depends_on"]["db"]["condition"] == "service_healthy"
    else:
        assert "depends_on" not in config["services"]["api"]
    assert set(config.get("volumes", {})) == (
        ({"db-data"} if database else set()) | ({"storage-data"} if storage else set()))


def test_the_pins_dependabot_reads_live_in_plain_compose_files():
    # Dependabot cannot read a .jinja file, so each pinned image sits in a plain compose file
    # that `directories` in .github/dependabot.yml reaches by glob.
    root = TEMPLATE.parent
    config = yaml.safe_load((root / ".github/dependabot.yml").read_text())
    entry = next(u for u in config["updates"] if u["package-ecosystem"] == "docker-compose")
    assert entry["directories"] == ["/template/backend/compose/*"]
    assert "directory" not in entry
    matched = sorted(p.parent.name for p in (TEMPLATE / "backend/compose").glob("*/compose.yaml"))
    assert matched == ["db", "storage"]
    for path in (TEMPLATE / "backend/compose").glob("*/compose.yaml"):
        for service in yaml.safe_load(path.read_text())["services"].values():
            assert re.search(r"@sha256:[0-9a-f]{64}$", service["image"]), path


def test_env_example_template_is_tracked_by_git_and_not_excluded():
    # `.env.*` is ignored by .gitignore and excluded in copier.yml so a developer's keys never
    # travel. .env.example must survive both, in its template form too: git ignoring
    # .env.example.jinja would leave it out of the commit and out of Copier's dirty-tree copy.
    root = TEMPLATE.parent
    ignored = subprocess.run(["git", "check-ignore", "-q", "template/backend/.env.example.jinja"],
                             cwd=root)
    assert ignored.returncode == 1, "template/backend/.env.example.jinja is git-ignored"
    config = yaml.safe_load((root / "copier.yml").read_text())
    assert ".env.*" in config["_exclude"] and "!.env.example" in config["_exclude"]


@ANSWERS
def test_make_targets_are_separated_by_one_blank_line(render, database, storage):
    # `{%- if %}` once ate the blank line before storage-up; the default output must read as before.
    mk = (render(database=database, storage=storage) / "mk/fastapi.mk").read_text()
    assert "\n\n\n" not in mk
    assert mk.endswith("\n") and not mk.endswith("\n\n")
    assert ("\n\nllm-report:" in mk) is database
    assert ("\n\nstorage-up:" in mk) is storage
    assert ("\n\nimage:" in mk)
