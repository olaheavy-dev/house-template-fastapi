"""The answers that switch parts of the backend on and off (#29, #30): `database`, `llm` and `storage`.

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
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from conftest import BASE_ANSWERS, TEMPLATE

# The valid triples (#30): `llm: langchain` needs the database, and `database: false` always means
# `llm: none`. Three shapes, each with storage on or off.
SHAPES = [(True, "langchain"), (True, "none"), (False, "none")]
TRIPLES = [(d, s, llm) for d, llm in SHAPES for s in (True, False)]
ANSWERS = pytest.mark.parametrize(
    ("database", "storage", "llm"), TRIPLES,
    ids=[f"database-{'on' if d else 'off'}-llm-{llm}-storage-{'on' if s else 'off'}"
         for d, s, llm in TRIPLES])

STORAGE_FILES = [
    "backend/src/app/storage.py",
    "backend/tests/storage.py",
    "backend/tests/test_storage.py",
    "backend/compose/storage/compose.yaml",
]
# The paths `llm: none` removes (and `database: false` with it), as patterns (fnmatch: `*` also
# crosses `/`). Each must match something that is removed, and everything removed must match one
# (test_llm_files_are_exact).
LLM_PATTERNS = [
    "backend/src/app/llm/*",
    "backend/src/app/prompts/*",
    "backend/llm-contracts.toml",
    "backend/tests/models.py",
    "backend/tests/recording.py",
    "backend/tests/test_llm_*.py",
    "backend/tests/test_recording_keys.py",
    "backend/tests/test_observability.py",
    "backend/tests/evals/*",
    "backend/tests/cassettes/*",
]
# One file from each pattern that must be absent without the gateway.
LLM_FILES = [
    "backend/src/app/llm/gateway.py",
    "backend/src/app/prompts/README.md",
    "backend/llm-contracts.toml",
    "backend/tests/models.py",
    "backend/tests/recording.py",
    "backend/tests/test_llm_gateway.py",
    "backend/tests/test_llm_errors.py",
    "backend/tests/test_recording_keys.py",
    "backend/tests/test_observability.py",
    "backend/tests/evals/conftest.py",
    "backend/tests/cassettes/test_llm_gateway/test_a_replayed_call_is_answered_and_its_cost_settled.yaml",
]
# The paths `database: false` removes besides the gateway's.
DATABASE_PATTERNS = [
    "backend/src/app/db.py",
    "backend/src/app/models.py",
    "backend/migrations/*",
    "backend/compose/db/*",
    "backend/tests/db.py",
    "backend/tests/test_db.py",
]
DATABASE_FILES = [
    "backend/src/app/db.py",
    "backend/src/app/models.py",
    "backend/migrations/env.py",
    "backend/migrations/versions/20261025_0001_enable_pgvector.py",
    "backend/migrations/versions/20261026_0002_llm_budget.py",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py",
    "backend/migrations/versions/20261026_0004_llm_call_details.py",
    "backend/migrations/versions/20261026_0005_per_user_llm_budget.py",
    "backend/migrations/versions/20261026_app_project_root.py",
    "backend/compose/db/compose.yaml",
    "backend/tests/db.py",
    "backend/tests/test_db.py",
]
# Files that exist in both renders (or in every render with the database) but change with the
# answer, each with an all-on plain twin: with the gateway, the rendered file equals the plain one.
TWINS = [
    "backend/src/app/main.py",
    "backend/src/app/settings.py",
    "backend/src/app/log.py",
    "backend/src/app/api/__init__.py",
    "backend/tests/conftest.py",
    "backend/.env.example",
    "backend/src/app/models.py",
    "backend/migrations/env.py",
    "backend/migrations/versions/20261026_0002_llm_budget.py",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py",
    "backend/migrations/versions/20261026_0004_llm_call_details.py",
    "backend/migrations/versions/20261026_0005_per_user_llm_budget.py",
]
# Words that must not appear in a storage-off project, except in pyproject.toml and uv.lock
# (the dependency stays declared: `boto3` in the extra, `types-boto3` in the dev group).
STORAGE_WORDS = re.compile(r"boto|STORAGE_|storage_settings|presigned|RustFS", re.IGNORECASE)
NOT_COMPARED = {"backend/pyproject.toml", "backend/uv.lock"}


def files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


@ANSWERS
def test_render_matrix(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    for path in DATABASE_FILES:
        assert (project / path).exists() is database, path
    for path in LLM_FILES:
        assert (project / path).exists() is (llm == "langchain"), path
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
    assert (answers["database"], answers["storage"], answers["llm"]) == (database, storage, llm)


@ANSWERS
def test_the_llm_import_contracts_exist_only_with_a_gateway(render, database, storage, llm):
    # The core's contract names app.llm.core, which import-linter refuses when the module is not
    # there, so it lives in its own file that exists only with the gateway (#33). The one
    # contract in pyproject.toml stays for every project (import-linter 2.16 exits 1 with none).
    gateway = llm == "langchain"
    project = render(database=database, storage=storage, llm=llm)
    contracts = project / "backend/llm-contracts.toml"
    assert contracts.exists() is gateway
    mk = (project / "mk/fastapi.mk").read_text()
    assert ("lint-imports --config llm-contracts.toml" in mk) is gateway
    pyproject = read_toml(project / "backend/pyproject.toml")
    assert len(pyproject["tool"]["importlinter"]["contracts"]) == 1
    if gateway:
        config = read_toml(contracts)["tool"]["importlinter"]
        assert config["root_package"] == "app"
        (contract,) = config["contracts"]
        assert contract["type"] == "forbidden"
        assert contract["source_modules"] == ["app.llm.core"]
        assert {"langchain", "langchain_core", "openai", "anthropic", "agent_framework"} <= set(
            contract["forbidden_modules"])


def render_other(render, tmp_path: Path, **answers) -> Path:
    """A second render in its own directory (the render fixture's own is `project`)."""
    dst = tmp_path / "other"
    dst.mkdir()
    (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump(BASE_ANSWERS))
    return render(dst, **answers)


def _exact(removed: set[str], patterns: list[str]) -> None:
    stray = [p for p in removed if not any(fnmatch.fnmatch(p, pat) for pat in patterns)]
    assert stray == []
    unused = [pat for pat in patterns if not any(fnmatch.fnmatch(p, pat) for p in removed)]
    assert unused == []


def test_llm_files_are_exact(render, tmp_path):
    # Everything `llm: none` removes (the database stays on) matches LLM_PATTERNS, and every
    # pattern removes something: a new file under a gateway path, or a pattern that matches
    # nothing, fails here.
    langchain = set(files(render(database=True, storage=True, llm="langchain")))
    none = set(files(render_other(render, tmp_path, database=True, storage=True, llm="none")))
    assert (none - langchain) == set()
    _exact(langchain - none, LLM_PATTERNS)


def test_database_files_are_exact(render, tmp_path):
    # `database: false` removes the gateway (llm none) and the database's own files: what is
    # removed beyond the gateway's matches DATABASE_PATTERNS, and every pattern removes something.
    none = set(files(render(database=True, storage=True, llm="none")))
    off = set(files(render_other(render, tmp_path, database=False, storage=True)))
    assert (off - none) == set()
    _exact(none - off, DATABASE_PATTERNS)


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
    # With every part on, the rendered .jinja file is byte-identical to the plain file the in-place
    # loop runs. Also: every .jinja next to a plain file is listed here (no unguarded twin) and
    # every listed twin really has both files.
    project = render(database=True, storage=True, llm="langchain")
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


DATABASE_NOT_SETTINGS = {"DB_PORT", "TEST_DATABASE_URL"}


@ANSWERS
def test_env_example_matches_settings(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    fields = {f.upper() for f in SETTING.findall((project / "backend/src/app/settings.py").read_text())}
    listed = set(EXAMPLE.findall((project / "backend/.env.example").read_text()))
    allowed = (NOT_SETTINGS - DATABASE_NOT_SETTINGS) | (DATABASE_NOT_SETTINGS if database else set()) | (
        STORAGE_NOT_SETTINGS if storage else set())
    assert fields - listed == set(), "settings missing from .env.example"
    assert listed - fields - allowed == set(), "variables in .env.example that are not settings"
    assert any(f.startswith("STORAGE_") for f in fields) is storage
    assert ("DATABASE_URL" in fields) is database
    # The model, embedding, key, agent and tracing settings exist only with the gateway.
    gateway = llm == "langchain"
    assert any(f.startswith(("LLM_", "LANGSMITH_", "AGENT_")) for f in fields) is gateway
    assert bool(fields & {"EMBEDDING_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}) is gateway
    assert bool(listed & {"EMBEDDING_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}) is gateway
    assert any(v.startswith(("LLM_", "LANGSMITH_", "AGENT_")) for v in listed) is gateway
    # The test database is the database's, not the gateway's.
    assert ("TEST_DATABASE_URL" in listed) is database


@ANSWERS
def test_production_guard_names_only_rendered_parts(render, database, storage, llm):
    settings = (render(database=database, storage=storage, llm=llm)
                / "backend/src/app/settings.py").read_text()
    assert ("DATABASE_URL is the local default" in settings) is database
    assert ("STORAGE_SECRET_ACCESS_KEY is the local default" in settings) is storage
    assert ("def no_local_defaults_in_production" in settings) is (database or storage)
    assert ("LOCAL_DATABASE_URL" in settings) is database
    assert ("model_validator" in settings) is (database or storage)


@ANSWERS
def test_conftest_fixtures_follow_the_answers(render, database, storage, llm):
    conftest = (render(database=database, storage=storage, llm=llm) / "backend/tests/conftest.py").read_text()
    tree = ast.parse(conftest)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    always = {"anyio_backend", "settings", "app", "client", "production_values",
              "expected_readiness_checks", "database_settings"}
    with_database = {"database_url", "db_session", "db_client"}
    with_llm = {"test_models", "live_runs_need_models_and_a_key", "vcr_config", "allowed_hosts",
                "is_eval_run", "is_recording"}
    with_storage = {"storage_endpoint", "storage_settings"}
    gateway = llm == "langchain"
    assert always <= defined
    assert (with_database <= defined) is database
    assert not (with_database & defined) or database
    assert (with_llm <= defined) is gateway
    assert not (with_llm & defined) or gateway
    assert (with_storage <= defined) is storage
    assert not (with_storage & defined) or storage
    if not gateway:
        # No LangSmith lines, no model names, no recording, and the budget's own sessionmaker is
        # not there for db_client to set.
        for word in ("LANGSMITH", "budget_sessionmaker", "CHAT_MODEL", "vcr", "OPENAI", "ANTHROPIC",
                     "SecretStr(os.environ"):
            assert word not in conftest, word
        assert "Settings(_env_file=None, app_env=\"test\")" in conftest
    if not database:
        assert "alembic" not in conftest.lower()
        assert 'return {}' in conftest  # production_values: nothing to set


MODEL_FRAMEWORKS = ("langchain", "langgraph", "langsmith", "openai", "anthropic")


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]


@pytest.mark.parametrize("storage", [True, False], ids=["storage-on", "storage-off"])
@pytest.mark.parametrize("database", [True, False], ids=["database-on", "database-off"])
def test_llm_none_imports_no_model_framework(render, database, storage):
    # Every rendered Python file, project-wide: nothing imports a model framework or provider SDK,
    # nor the gateway's own modules (app.llm, app.prompts, tests.models, tests.recording).
    project = render(database=database, storage=storage, llm="none")
    gateway = ("app.llm", "app.prompts", "tests.models", "tests.recording")
    for rel, path in files(project).items():
        if path.suffix != ".py":
            continue
        for module in _imported_modules(path):
            assert not module.startswith(MODEL_FRAMEWORKS), (rel, module)
            assert not module.startswith(gateway), (rel, module)
        # `from app import llm` style imports too.
        assert not re.search(r"from app import [^\n]*\b(llm|prompts)\b", path.read_text()), rel
        assert "from tests import" not in path.read_text() or not re.search(
            r"from tests import [^\n]*\b(models|recording)\b", path.read_text()), rel


@pytest.mark.parametrize("database", [True, False], ids=["database-on", "database-off"])
def test_settings_demand_no_model_when_llm_is_none(render, database):
    settings = (render(database=database, llm="none") / "backend/src/app/settings.py").read_text()
    for word in ("llm_model", "embedding_model", "api_key", "agent_", "AGENT_MAX_STEPS",
                 "no_removed_agent_max_steps", "langsmith", "Decimal", "Any"):
        assert word not in settings, word
    # `extra="ignore"` keeps a leftover LLM_MODEL in backend/.env or the environment harmless.
    assert 'extra="ignore"' in settings


@ANSWERS
def test_no_rendered_python_file_has_an_unused_import(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    for rel, path in files(project).items():
        if path.suffix != ".py" or not rel.startswith(
                ("backend/src/", "backend/tests/", "backend/migrations/")):
            continue
        text = path.read_text()
        tree = ast.parse(text)
        lines = text.splitlines()
        imported = {}
        for node in ast.walk(tree):
            # An import kept on purpose says so (`# noqa: F401`, as the migrations' pgvector one).
            if isinstance(node, (ast.Import, ast.ImportFrom)) and "noqa: F401" in lines[node.lineno - 1]:
                continue
            if isinstance(node, ast.Import):
                imported.update({(a.asname or a.name).split(".")[0]: node for a in node.names})
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                imported.update({a.asname or a.name: node for a in node.names})
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        exported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(x, "id", "") == "__all__" for x in node.targets):
                exported = {e.value for e in node.value.elts}
        unused = sorted(set(imported) - used - exported)
        # `from app import api, db` style imports of modules count as used when the name is used.
        assert unused == [], (rel, unused)


# Template-owned files that exist only with the gateway, and only with the database (the project's
# own tests are not listed). The migrations stay with the database: the gateway's four are empty
# twins without the gateway, and still the template's.
LLM_OWNED = [
    "backend/src/app/llm/__init__.py",
    "backend/src/app/llm/budget.py",
    "backend/src/app/llm/checkpoints.py",
    "backend/src/app/llm/gateway.py",
    "backend/src/app/llm/embeddings.py",
    "backend/src/app/llm/errors.py",
    "backend/src/app/llm/metrics.py",
    "backend/src/app/llm/prompts.py",
    "backend/src/app/llm/report.py",
    "backend/src/app/llm/core/*.py",
    "backend/src/app/llm/langchain/*.py",
    "backend/llm-contracts.toml",
    "backend/tests/evals/conftest.py",
]
DATABASE_OWNED = [
    "backend/migrations/env.py",
    "backend/migrations/script.py.mako",
    "backend/migrations/versions/20261025_0001_enable_pgvector.py",
    "backend/migrations/versions/20261026_0002_llm_budget.py",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py",
    "backend/migrations/versions/20261026_0004_llm_call_details.py",
    "backend/migrations/versions/20261026_0005_per_user_llm_budget.py",
    "backend/migrations/versions/20261026_app_project_root.py",
    "backend/compose/db/compose.yaml",
    "backend/tests/db.py",
]


@ANSWERS
def test_every_template_owned_entry_exists(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    owned = (project / ".github/template-owned/fastapi.txt").read_text().splitlines()
    patterns = [ln for ln in owned if ln and not ln.startswith("#")]
    existing = list(files(project))
    for pattern in patterns:
        assert any(fnmatch.fnmatch(f, pattern) for f in existing), pattern
    # And the reverse for the parts that switch: a file is listed exactly when it exists.
    for path in LLM_OWNED:
        assert (path in patterns) is (llm == "langchain"), path
    for path in DATABASE_OWNED:
        assert (path in patterns) is database, path
    for path in STORAGE_FILES:
        if path != "backend/tests/test_storage.py":  # the project's own tests are not listed
            assert (path in patterns) is storage, path


def _owned_patterns(project: Path) -> list[str]:
    owned = (project / ".github/template-owned/fastapi.txt").read_text().splitlines()
    return [ln for ln in owned if ln and not ln.startswith("#")]


def _is_owned(patterns: list[str], path: str) -> bool:
    # The base project's checker uses fnmatch, whose `*` also matches `/`.
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def test_every_llm_module_but_prices_and_checkpoint_types_and_every_template_migration_is_template_owned(render):
    # The warning shims (errors, embeddings, ...) and the template's own migrations are the
    # template's (an edit to one would conflict on the next `copier update`); listing them makes
    # the project's CI warn when a pull request edits one. prices.py is the project's price
    # table and checkpoint_types.py the project's list of its own saved classes (#36), so both stay
    # off the list.
    project = render(database=True, storage=True)
    patterns = _owned_patterns(project)
    llm = sorted(f for f in files(project)
                 if f.startswith("backend/src/app/llm/") and f.endswith(".py"))
    assert len(llm) > 20  # the prefix found the modules
    versions = sorted(f"backend/migrations/versions/{v.name}"
                      for v in (TEMPLATE / "backend/migrations/versions").glob("*.py"))
    # Five numbered migrations and the project-root one; `>=` still proves the glob found them.
    # Update the owned list together with any new template migration.
    assert len(versions) >= 6
    yours = {"backend/src/app/llm/prices.py", "backend/src/app/llm/checkpoint_types.py"}
    not_owned = [f for f in llm if f not in yours and not _is_owned(patterns, f)]
    not_owned += [v for v in versions if v not in patterns]  # by exact path
    assert not_owned == []
    # Spelled out in full: a `versions/*` glob would also capture the project's own migrations.
    for mine in ("backend/migrations/versions/20270101_abcd1234_mine.py",
                 "backend/src/app/llm/prices.py", "backend/src/app/llm/checkpoint_types.py",
                 "backend/src/app/api/mine.py"):
        assert not _is_owned(patterns, mine), mine


@ANSWERS
def test_rendered_project_has_no_leftover_template_syntax(render, database, storage, llm):
    for path in files(render(database=database, storage=storage, llm=llm)).values():
        if path.suffix in {".py", ".toml", ".md", ".mk", ".yml", ".yaml", ".txt", ".example"}:
            # GitHub Actions expressions (${{ ... }}) are meant to be there.
            assert not re.search(r"(?<!\$)\{\{|\{%", path.read_text()), path


@ANSWERS
def test_rendered_text_files_end_in_exactly_one_newline(render, database, storage, llm):
    # The base project's pre-commit end-of-file fixer fails `make check` on a file that ends in a
    # blank line or has no final newline (a `{% endif %}` after the last line once did).
    # Every text file, whatever its name (Dockerfile, .dockerignore, .python-version included).
    for rel, path in files(render(database=database, storage=storage, llm=llm)).items():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        assert not text or (text.endswith("\n") and not text.endswith("\n\n")), rel


@ANSWERS
def test_every_rendered_python_file_parses(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    for rel, path in files(project).items():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=rel)


# Words that must not appear in a database-less project, except in pyproject.toml and uv.lock (the
# packages stay declared in the extras).
LLM_WORDS = re.compile(
    r"langchain|langgraph|langsmith|LLM_MODEL|EMBEDDING_MODEL|openai|anthropic|gateway|llm-report|"
    r"record-llm|make eval|cassette|AGENT_MAX", re.IGNORECASE)
# "checkpoint" is not a word to look for: migrations/env.py names the checkpoint tables it ignores.
DATABASE_WORDS = re.compile(
    r"sqlalchemy|alembic|DATABASE_URL|db_session|db_client|SessionDep|pgvector|psycopg|db-up|"
    r"make migrat", re.IGNORECASE)
# The sentences the rules file keeps whatever the answers, because they tell a project what it could
# switch on. They are the only allowances, and each must be there exactly once (a stale allowance
# would hide nothing, a duplicated one could hide a leftover).
EXTRAS_SENTENCE = ("`db` comes with the `database` answer, `langchain` with `llm: langchain`, "
                   "`storage` with the `storage` answer")
# The command that turns the gateway on. With no database it only adds to the database's own.
SWITCH_ON_WITH_DATABASE = "(add `-d llm=langchain` for the model gateway too: it needs the database)"
STAYS_OFF = "(the gateway stays off once the answers file records `llm`, which the first 0.8.0 update does)"
SWITCH_ON = "`-d llm=langchain` to turn the model gateway on"
# For a database that ran without the gateway: its migrations did nothing, so the tables are missing.
RECIPE = ("A database that ran without the model tables (migrations 0002 to 0005 did nothing there) "
          "gets them only after `copier update` has turned the model gateway on (stamping first would "
          "make `make migrate` re-run the empty revisions and leave no tables). Then, on every database "
          "the project runs against, production and staging included, and before the gateway is "
          "deployed, run `uv --directory backend run --locked --extra db --extra langchain alembic "
          "stamp 0001`, then `make migrate`; a new database needs only `make migrate`.")
ALLOWED = {
    "database": {".claude/rules/fastapi.md": [EXTRAS_SENTENCE, STAYS_OFF, SWITCH_ON_WITH_DATABASE]},
    "llm": {".claude/rules/fastapi.md": [EXTRAS_SENTENCE, SWITCH_ON, RECIPE]},
    "storage": {},
}
PARTS = {"storage": (STORAGE_WORDS, "boto3", ["backend/src/app/storage.py", "mk/fastapi.mk"]),
         "database": (re.compile(DATABASE_WORDS.pattern + "|" + LLM_WORDS.pattern, re.IGNORECASE),
                      "sqlalchemy", ["backend/src/app/main.py", "mk/fastapi.mk"]),
         "llm": (LLM_WORDS, "langchain", ["backend/src/app/main.py", "mk/fastapi.mk"])}
# What the answers are for each part switched off (the others stay on, as in a default render).
OFF = {"storage": {"storage": False}, "database": {"database": False}, "llm": {"llm": "none"}}


def word_hits(project: Path, words: re.Pattern[str], part: str, allow: bool = True) -> list[tuple[str, str]]:
    found = []
    for rel, path in files(project).items():
        if rel in NOT_COMPARED or ".git/" in rel:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for phrase in ALLOWED[part].get(rel, []) if allow else []:
            assert text.count(phrase) == 1, (rel, phrase)
            text = text.replace(phrase, "")
        found += [(rel, m.group(0)) for m in words.finditer(text)]
    return found


@ANSWERS
def test_rules_switch_names_every_extra_and_the_flag_that_changes_each_part(
        render, database, storage, llm):
    rules = (render(database=database, storage=storage, llm=llm) / ".claude/rules/fastapi.md").read_text()
    bullet = next(ln for ln in rules.splitlines() if ln.startswith("- Dependencies:"))
    assert EXTRAS_SENTENCE in bullet
    gateway = llm == "langchain"
    for part, on in (("database", database), ("storage", storage), ("llm", gateway)):
        # The command turns the part to the other state: off when it is on, on when it is off.
        off_flag = "-d llm=none" if part == "llm" else f"-d {part}=false"
        on_flag = "-d llm=langchain" if part == "llm" else f"-d {part}=true"
        assert f"{off_flag if on else on_flag}`" in bullet, part
        assert f"{on_flag if on else off_flag}`" not in bullet, part
    # Without the database there is no gateway to turn off on its own, and turning it on needs both.
    if not database:
        assert "-d llm=none`" not in bullet
        assert SWITCH_ON_WITH_DATABASE in bullet and STAYS_OFF in bullet
    # The recipe for a database that ran without the gateway appears only where that can be the case.
    assert (RECIPE in bullet) is (database and not gateway)


@pytest.mark.parametrize("part", ["storage", "database", "llm"], ids=["storage-off", "database-off", "llm-off"])
def test_no_part_left_behind(render, part):
    words, dependency, _ = PARTS[part]
    project = render(**OFF[part])
    assert word_hits(project, words, part) == []
    # The mentions that remain are only the declared dependency.
    pyproject = (project / "backend/pyproject.toml").read_text()
    assert dependency in pyproject


@pytest.mark.parametrize("part", ["storage", "database", "llm"])
def test_no_part_left_behind_check_catches_a_leftover(render, part):
    # The word list does find what it is meant to find (a check that cannot fail proves nothing).
    words, _, must_hit = PARTS[part]
    project = render()
    hits = {rel for rel, _ in word_hits(project, words, part, allow=False)}
    assert set(must_hit) <= hits
    assert {".claude/rules/fastapi.md"} <= hits
    if part != "llm":
        assert "backend/compose.yaml" in hits


def extras_flags(database: bool, storage: bool, llm: str) -> str:
    # `langchain` always comes with `db` (the gateway keeps its budget and checkpoints in PostgreSQL).
    return " ".join(
        (["--extra db"] if database else []) + (["--extra langchain"] if llm == "langchain" else [])
        + (["--extra storage"] if storage else []))


@ANSWERS
def test_the_image_installs_the_answered_extras(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    flags = extras_flags(database, storage, llm)
    gateway = llm == "langchain"
    dockerfile = (project / "backend/Dockerfile").read_text()
    syncs = [ln for ln in dockerfile.splitlines() if "uv sync" in ln]
    assert len(syncs) == 2
    for ln in syncs:
        assert ln.endswith(" ".join(["--no-editable", flags]).strip()), ln
    # The shutdown comment names the model timeout only where there is a model timeout.
    assert ("LLM_TIMEOUT_SECONDS" in dockerfile) is gateway
    mk = (project / "mk/fastapi.mk").read_text()
    runs = [ln for ln in mk.splitlines() if "uv --directory backend" in ln and " sync " in ln
            or ln.startswith(("BACKEND :=", "LIVE :="))]
    # Every uv call that installs or runs the project goes through EXTRAS, set from the answers.
    # LIVE (recording and evals) exists only with the gateway.
    assert len(runs) == (3 if gateway else 2) and all("$(EXTRAS)" in ln for ln in runs), runs
    assert ("LIVE :=" in mk) is gateway
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
        # Worked out from the plan: the database without the gateway installs db (and storage) only.
        if (database, llm) == (True, "none"):
            assert all("--locked --extra db" in ln and "langchain" not in ln for ln in lines)
    # `make lock` locks every extra whatever the answers (the lockfile is not per answer).
    assert "uv --directory backend lock\n" in mk


ALWAYS_TARGETS = {"install-fastapi", "check-fastapi", "check-fast-fastapi", "lock", "dev", "test",
                  "image", "up"}
DATABASE_TARGETS = {"db-up", "migrate", "migration"}
LLM_TARGETS = {"llm-report", "record-llm", "eval"}


@ANSWERS
def test_make_targets_follow_the_answers(render, database, storage, llm):
    gateway = llm == "langchain"
    mk = (render(database=database, storage=storage, llm=llm) / "mk/fastapi.mk").read_text()
    want = (ALWAYS_TARGETS | (DATABASE_TARGETS if database else set())
            | (LLM_TARGETS if gateway else set()) | ({"storage-up"} if storage else set()))
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
    # The model test run unsets LLM_EVAL, which only the gateway's evals read.
    assert ("env -u LLM_EVAL" in mk) is gateway
    assert ("comma := ," in mk) is gateway
    if not gateway:
        assert not any(word in mk for word in ("comma", "record-llm", "LLM_", "llm-contracts"))
    if not database:
        assert "alembic" not in mk


@ANSWERS
def test_plain_mypy_reads_only_paths_every_project_has(render, database, storage, llm):
    # A plain `uv run mypy` reads `files` from pyproject.toml, which is not per answer: a path that
    # is missing is an error ("Cannot read file"), so only the make target adds migrations/.
    project = render(database=database, storage=storage, llm=llm)
    paths = read_toml(project / "backend/pyproject.toml")["tool"]["mypy"]["files"]
    missing = [p for p in paths if not (project / "backend" / p).exists()]
    assert missing == []


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
def test_compose_services_follow_the_answers(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
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
def test_make_targets_are_separated_by_one_blank_line(render, database, storage, llm):
    # `{%- if %}` once ate the blank line before storage-up; the default output must read as before.
    mk = (render(database=database, storage=storage, llm=llm) / "mk/fastapi.mk").read_text()
    assert "\n\n\n" not in mk
    assert mk.endswith("\n") and not mk.endswith("\n\n")
    assert ("\n\nllm-report:" in mk) is (llm == "langchain")
    assert ("\n\ndb-up:" in mk) is database
    assert ("\n\nstorage-up:" in mk) is storage
    assert ("\n\nimage:" in mk)


# The `llm` answer (#30): none or langchain; langchain needs the database. database: false means none.
@pytest.mark.parametrize(
    ("database", "llm"), [(True, "langchain"), (False, "none")], ids=["database-on", "database-off"])
def test_llm_defaults_follow_the_database(render, database, llm):
    # Worked out from the plan: the default is `langchain` if database else `none`, recorded in the
    # answers file even when it is the default (no `when` on the question).
    project = render(database=database)
    answers = yaml.safe_load((project / ".copier-answers.fastapi.yml").read_text())
    assert answers["llm"] == llm


def test_llm_without_database_is_refused(render):
    with pytest.raises(ValueError, match="langchain needs database: true"):
        render(database=False, llm="langchain")


def test_llm_none_is_allowed_with_the_database(render):
    project = render(database=True, llm="none")
    answers = yaml.safe_load((project / ".copier-answers.fastapi.yml").read_text())
    assert (answers["database"], answers["llm"]) == (True, "none")


@pytest.mark.parametrize("llm", ["maf", "LangChain", "NONE", "true"])
def test_llm_refuses_an_unknown_framework(render, llm):
    # `maf` arrives with #35; answers are case-sensitive.
    with pytest.raises(ValueError, match="Invalid choice for 'llm'"):
        render(database=True, llm=llm)


# The migration chain (#30). Worked out from the files the template ships (the template's revisions
# 0001 to 0005 on the "template" branch, the project's root on the "app" branch), not from a run.
CHAIN = {"0001": None, "0002": "0001", "0003": "0002", "0004": "0003", "0005": "0004",
         "app_root": None}
GATEWAY_REVISIONS = {"0002", "0003", "0004", "0005"}


def _migrations(project: Path) -> dict[str, dict]:
    """Each migration of a render, read with ast only (nothing is imported or run)."""
    found = {}
    for path in sorted((project / "backend/migrations/versions").glob("*.py")):
        tree = ast.parse(path.read_text())
        values = {}
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                values[node.target.id] = ast.literal_eval(node.value)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        found[values["revision"]] = {
            "down": values["down_revision"], "labels": values["branch_labels"], "imports": imports,
            "empty": all(len(functions[f].body) == 1 and isinstance(functions[f].body[0], ast.Pass)
                         for f in ("upgrade", "downgrade")),
        }
    return found


@ANSWERS
def test_migration_chain_is_linear_in_every_variant(render, database, storage, llm):
    project = render(database=database, storage=storage, llm=llm)
    if not database:
        assert not (project / "backend/migrations").exists()
        return
    migrations = _migrations(project)
    # The same revision ids and parents whatever the gateway answer: a project that ran the
    # gateway's migrations keeps a chain Alembic can follow.
    assert {rev: m["down"] for rev, m in migrations.items()} == CHAIN
    assert migrations["0001"]["labels"] == ("template",)
    assert migrations["app_root"]["labels"] == ("app",)
    assert sorted(rev for rev, m in migrations.items() if m["down"] is None) == ["0001", "app_root"]
    for rev in ("0002", "0003", "0004", "0005"):
        assert migrations[rev]["labels"] is None, rev
    gateway = llm == "langchain"
    for rev in GATEWAY_REVISIONS:
        # Without the gateway the revision stays and does nothing: no table, no app.llm import.
        assert migrations[rev]["empty"] is not gateway, rev
        if not gateway:
            assert not [i for i in migrations[rev]["imports"] if i.startswith("app.")], rev
    for rev in ("0001", "app_root"):
        assert migrations[rev]["empty"] is (rev == "app_root"), rev


def _string_set(tree: ast.AST, name: str) -> set[str]:
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == name:
            value = node.value
        if value is not None:
            if isinstance(value, ast.Call):  # frozenset({...})
                value = value.args[0]
            return {e.value for e in value.elts}
    raise AssertionError(f"{name} not found")


def test_llm_none_autogenerate_ignores_every_table_the_gateway_made(render, tmp_path):
    # A project that ran the gateway and then turned it off keeps its tables; autogenerate must not
    # propose dropping them. The set env.py ignores equals the tables the gateway's migrations
    # create plus the library's own four.
    langchain = render(database=True, storage=True, llm="langchain")
    created = set()
    for migration in (langchain / "backend/migrations/versions").glob("*.py"):
        for node in ast.walk(ast.parse(migration.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "create_table"):
                created.add(node.args[0].value)
    checkpoint_tables = _string_set(
        ast.parse((langchain / "backend/src/app/llm/langchain/checkpoints.py").read_text()), "TABLES")
    assert created == {"llm_call", "llm_spend_day", "llm_spend_user_day"}
    assert checkpoint_tables == {"checkpoint_migrations", "checkpoints", "checkpoint_blobs",
                                 "checkpoint_writes"}
    none = render_other(render, tmp_path, database=True, storage=True, llm="none")
    env = ast.parse((none / "backend/migrations/env.py").read_text())
    assert _string_set(env, "LEFTOVER_TABLES") == created | checkpoint_tables
    # And the version with the gateway has no such list (it imports the library's own).
    assert "LEFTOVER_TABLES" not in (langchain / "backend/migrations/env.py").read_text()
