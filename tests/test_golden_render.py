"""Safety net for template changes: the default render may only change on purpose.

Docs: Copier 9.19.1 (the version pinned in uv.lock)
- Python API (`run_copy`, `run_update`): https://copier.readthedocs.io/en/v9.19.1/reference/api/
- Updating: https://copier.readthedocs.io/en/v9.19.1/updating/

How it works
- BASELINE_REF is a released (or agreed) state of this template. The default render at that ref
  is compared, file by file, with the default render of the working tree.
- A feature PR that changes the default output on purpose adds `path: "reason (#NN)"` to INTENDED.
- After each release a `chore:` PR moves BASELINE_REF to the new tag and empties INTENDED.
- No snapshot is stored: both renders are made fresh, so there is nothing to keep in step.
"""

from __future__ import annotations

import difflib
import subprocess
from collections.abc import Mapping
from pathlib import Path

import copier
import pytest
import yaml

from conftest import BASE_ANSWERS, ROOT

# main at d67ca38 ("no default models or prices", #42), the state #29 must not change by accident.
BASELINE_REF = "d67ca3807ae74a78cc9728bf20da829be035117c"

# {rendered path: "reason (#NN)"}. Starts empty. The test fails for a path that differs without an
# entry here, and for an entry whose path no longer differs. A path-level entry hides a second,
# accidental change to the same file until the next baseline move.
# Not compared: file modes, symlinks and empty directories (the template has no tracked
# executables today).
INTENDED: dict[str, str] = {
    ".copier-answers.fastapi.yml": "records container_images (#49), storage and database (#29)",
    ".claude/rules/fastapi.md": "rule text moved after backend/.env (#41); extras rule naming db, langchain and storage, the switch command for each answer, turning the database off, make check type-checks migrations/, 413 sentence moved out of the Files bullet into its own (#29); says where the price arithmetic lives and that make check runs both lint-imports commands (#33); says where model-call failures are classified (#33); says where the budget, its Meter, the bounds and the metrics live (#33); says where the OpenAI embeddings and their batching live and that an embedding price's max_prompt_tokens applies to each request (#33); says where the prompt and its LangChain config live and that the report is python -m app.llm.core.report (#33); says where the gateway and the checkpointer live and that GatewayDep is imported from app.llm (#33)",
    "mk/fastapi.mk": "record-llm help says provider keys (#41); EXTRAS, compose path, mypy paths (#29); check-fastapi also runs lint-imports --config llm-contracts.toml with the database (#33); llm-report runs python -m app.llm.core.report (#33)",
    "backend/src/app/llm/prices.py": "comment matches the UTC+1 date rule (#41); keeps only the price tables, the arithmetic moved to app/llm/core/prices.py and the old names warn until 0.8.0 (#33)",
    "backend/tests/test_llm_prices.py": "cache_creation ceiling case, UTC+1 date rule (#41); imports from app/llm/core (#33)",
    ".github/template-owned/fastapi.txt": "lists the compose files under compose/ (#29); lists backend/.python-version (#59); lists the llm core and langchain adapter files and llm-contracts.toml (#33)",
    "backend/.python-version": "uv stops looking for .python-version at the project root, so backend/ needs its own copy (#59)",
    "backend/Dockerfile": "installs the answered extras; the uvicorn comment is a whole sentence (#29)",
    "backend/compose.yaml": "includes the db and storage services from compose/ (#29)",
    "backend/compose/db/compose.yaml": "the db service moved out of compose.yaml (#29)",
    "backend/compose/storage/compose.yaml": "the storage service moved out of compose.yaml (#29)",
    "backend/pyproject.toml": "boto3, database and gateway packages moved to the storage, db and langchain extras; plain mypy reads src and tests only (#29); pytest ignores langsmith's Python 3.14 asyncio warning (#59); the gateway import contract ignores app.llm.core.openai_embeddings -> openai (#33)",
    "backend/uv.lock": "locks the db, langchain and storage extras (#29)",
    "backend/src/app/storage.py": "docstring names compose/storage/compose.yaml (#29)",
    "backend/src/app/limits.py": "docstring no longer names storage (#29)",
    "backend/src/app/settings.py": "comment lines re-wrapped for the storage switch, compose paths in comments (#29)",
    "backend/tests/__init__.py": "BACKEND moved here so storage.py can be excluded (#29)",
    "backend/tests/db.py": "BACKEND from tests, db image from compose/db (#29)",
    "backend/tests/storage.py": "BACKEND from tests, image from compose/storage (#29)",
    "backend/tests/test_hardening.py": "storage secret check moved to test_storage.py; database and pool tests moved out, production_values (#29)",
    "backend/tests/test_storage.py": "gains the production storage secret check; production_values, database_settings (#29)",
    "backend/.dockerignore": "compose/ stays out of the image (#29)",
    "backend/tests/test_db.py": "BACKEND from tests; gains the database tests moved out of test_hardening, readiness check test (#29)",
    "backend/tests/test_health.py": "neutral check name, readiness without storage, no-check case, expected_readiness_checks (#29)",
    "backend/tests/conftest.py": "recording checks the key of both pinned models (#41); production_values, expected_readiness_checks, database_settings (#29)",
    "backend/src/app/health.py": "docstring no longer names the database (#29)",
    "backend/src/app/auth.py": "docstring example and id note name no database or gateway (#29)",
    "backend/tests/test_app.py": "production settings from production_values (#29)",
    "backend/tests/test_auth.py": "production settings from production_values (#29)",
    "backend/tests/test_errors.py": "model-failure tests moved to test_llm_errors.py; the log test uses the httpx logger (#29)",
    "backend/tests/test_llm_errors.py": "model-failure and budget-refusal tests moved out of test_errors.py (#29); BudgetExhausted from app/llm/core/budget.py (#33)",
    "backend/tests/test_observability.py": "request-trace test moved to test_telemetry.py (#29); price arithmetic from app/llm/core (#33); imports from app/llm/core/budget.py and langchain/callbacks.py (#33); the report comes from app/llm/core/report.py (#33)",
    "backend/tests/test_telemetry.py": "the request-trace test moved out of test_observability.py (#29)",
    "backend/tests/test_llm_budget.py": "gains the budget-pool test moved out of test_hardening.py (#29); a class in two error families settles by its first kind (#33); imports from app/llm/core and app/llm/langchain/callbacks.py (#33); the report comes from app/llm/core/report.py (#33)",
    "backend/tests/test_llm_cost_pins.py": "characterises the cost arithmetic before the framework-neutral split (#33)",
    "backend/tests/test_llm_surface.py": "characterises the gateway's import surface, failure mapping, metrics and prompts before the framework-neutral split (#33); error names imported from app/llm/core and app/llm/langchain, and listed with their new modules (#33); budget and metrics names listed with their new modules; aliases convert LangChain usage (#33); the old embeddings module's budget names are aliased from their new modules (#33); the embeddings class and openai_client are listed as moved, with the new modules' public names (#33); the prompt names are listed with their new modules (#33); the report is imported from app/llm/core/report.py and its names listed as moved (#33); the checkpointer names are listed as moved (#33); the gateway names are listed as moved (#33)",
    "backend/tests/test_llm_ledger.py": "characterises the budget ledger, reservation and spend report before the framework-neutral split (#33); the embeddings seam note names _KINDS and REJECTED (#33); imports from app/llm/core and app/llm/langchain/callbacks.py; settle alias converts LangChain usage (#33); GatewayEmbeddings comes from app/llm/langchain/embeddings.py and the embeddings seam note names the core's KINDS (#33); the report comes from app/llm/core/report.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/src/app/llm/budget.py": "prices the settled call with the neutral Usage (#33); a failed call's refund is decided by its core failure kind (#33); moved to app/llm/core/budget.py, core/bounds.py and langchain/callbacks.py; the old names warn until 0.8.0 (#33)",
    "backend/src/app/llm/embeddings.py": "prices embeddings with the neutral Usage (#33); OpenAI errors classified as core failure kinds (#33); budget, bounds and metrics from app/llm/core, settled with the neutral Usage (#33); batches and meters on a Meter through app/llm/core/embeddings.py and calls OpenAI through core/openai_embeddings.py, honouring an embedding price's max_prompt_tokens (#33); moved to app/llm/langchain/embeddings.py and app/llm/core/openai_embeddings.py; the old names warn until 0.8.0 (#33)",
    "backend/src/app/llm/metrics.py": "token_details comes from app/llm/core/prices.py (#33); moved to app/llm/core/metrics.py; the old names warn until 0.8.0 (#33)",
    "backend/src/app/llm/report.py": "price arithmetic comes from app/llm/core/prices.py (#33); the ledger comes from app/llm/core/budget.py (#33); moved to app/llm/core/report.py; the old names warn until 0.8.0, and python -m app.llm.report still runs the report, with a warning (#33)",
        "backend/tests/evals/conftest.py": "key_setting comes from app/llm/core/registry.py (#33); the checkpointer comes from app/llm/langchain/checkpoints.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/tests/test_llm_shims.py": "new: the old import paths still answer, with a warning, and the template never uses them (#33); app.llm.errors too (#33); app.llm.budget and app.llm.metrics too; the tables are mapped once (#33); app.llm.embeddings too (#33); app.llm.prompts too (#33); app.llm.report too (#33); app.llm.checkpoints too, and the template's migrations never use an old path (#33); the gateway names, GatewayDep as one object, app.llm loading no framework and not shadowing submodules (#33)",
    "backend/tests/test_llm_gateway.py": "the cost check goes through app/llm/core/prices.py and from_langchain (#33); LlmCall from app/llm/core/budget.py (#33); load_prompt from app/llm/langchain/prompts.py (#33); Gateway from app/llm/langchain/gateway.py and GatewayDep from app.llm (#33)",
    "backend/src/app/llm/gateway.py": "key_setting and the price lookups come from app/llm/core/registry.py; the old names warn until 0.8.0 (#33); BudgetGuard comes from app/llm/langchain/callbacks.py (#33); GatewayEmbeddings and openai_client come from their new modules (#33); Prompt comes from app/llm/core/prompts.py and call_config builds the config through prompt_config and Prompt.metadata() (#33); moved to app/llm/langchain/gateway.py, GatewayDep to app.llm.GatewayDep; the old names warn until 0.8.0 (#33)",
    "backend/llm-contracts.toml": "new: import contract keeping app/llm/core free of model frameworks, only with the database (#33); ignores the one import of openai in app/llm/core/openai_embeddings.py (#33)",
    "backend/src/app/llm/__init__.py": "docstring says prices.py holds the price tables, core/ prices calls and classifies their failures and langchain/ converts (#33); docstring says core/ holds the budget and metrics (#33); names the metered embeddings (#33); names where prompts live (#33); answers GatewayDep lazily, and says the core must not import the package (#33)",
    "backend/src/app/llm/core/__init__.py": "new: the framework-neutral core (#33); names prompts.py and report.py (#33); names errors.py (#33); names bounds.py, budget.py and metrics.py (#33); names embeddings.py and openai_embeddings.py, the one core module that imports a provider SDK (#33)",
    "backend/src/app/llm/core/usage.py": "new: the neutral Usage type (#33)",
    "backend/src/app/llm/core/prices.py": "new: the price arithmetic moved out of app/llm/prices.py (#33); the max_prompt_tokens comment says an embedding price bounds each request (#33)",
    "backend/src/app/llm/core/registry.py": "new: provider-key and price lookup moved out of the gateway (#33)",
    "backend/src/app/llm/langchain/__init__.py": "new: the LangChain adapter package (#33); converts model errors too (#33); names callbacks.py (#33); names embeddings.py (#33); names prompts.py, the LangChain call config (#33); names gateway.py and checkpoints.py (#33)",
    "backend/src/app/llm/langchain/usage.py": "new: from_langchain converts LangChain usage to the neutral Usage (#33)",
    "backend/tests/test_llm_core.py": "new: tests of the neutral core (#33); the core may import openai in openai_embeddings.py only (#33)",
    "backend/tests/test_llm_usage.py": "new: tests of from_langchain (#33)",
    "backend/tests/test_llm_error_logs.py": "new: characterises what a model failure logs (#33)",
    "backend/src/app/llm/core/errors.py": "new: the framework-neutral failure kinds, outcomes and handlers (#33)",
    "backend/tests/test_llm_core_errors.py": "new: tests of the neutral failure family (#33)",
    "backend/src/app/main.py": "installs the core's failure handlers and LangChain's from their new modules (#33); imports the checkpointer from app/llm/langchain/checkpoints.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/src/app/llm/langchain/errors.py": "new: LangChain's model errors as core failure kinds, and their handlers (#33)",
    "backend/tests/test_llm_langchain_errors.py": "new: tests of LangChain's error classification (#33)",
    "backend/tests/test_llm_embeddings.py": "comment names app/llm/langchain/errors.py (#33); budget names from app/llm/core/budget.py (#33); GatewayEmbeddings comes from app/llm/langchain/embeddings.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/src/app/llm/errors.py": "moved to app/llm/core/errors.py and app/llm/langchain/errors.py; the old names warn until 0.8.0 (#33)",
    "backend/tests/recording.py": "recording key check, importable by its test (#41); key_setting from app/llm/core/registry.py (#33)",
    "backend/tests/test_recording_keys.py": "tests for the recording key check (#41)",
    "backend/src/app/models.py": "the llm tables come from app/llm/core/budget.py (#33)",
    "backend/tests/test_llm_agents.py": "LlmCall from app/llm/core/budget.py (#33); Prompt from app/llm/langchain/prompts.py (#33); the checkpointer comes from app/llm/langchain/checkpoints.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/tests/evals/test_gateway_evals.py": "budget names from app/llm/core/budget.py (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/src/app/llm/core/bounds.py": "new: the request bounds moved out of budget.py (#33)",
    "backend/src/app/llm/core/budget.py": "new: the ledger and the Meter moved out of budget.py (#33); Meter.abandon forgets a cancelled call (#33); expire_stale names python -m app.llm.core.report (#33)",
    "backend/src/app/llm/core/metrics.py": "new: the llm metrics moved out of metrics.py, recorded from the neutral Usage (#33)",
    "backend/src/app/llm/langchain/callbacks.py": "new: BudgetGuard as a thin layer over the Meter (#33)",
    "backend/src/app/llm/langchain/embeddings.py": "new: GatewayEmbeddings moved out of app/llm/embeddings.py, a thin layer over the core's metered embeddings (#33)",
    "backend/tests/test_llm_core_budget.py": "new: tests of the neutral budget, bounds and metrics (#33)",
    "backend/src/app/llm/core/embeddings.py": "new: embeddings batched and metered on a Meter, for any framework (#33)",
    "backend/src/app/llm/core/openai_embeddings.py": "new: OpenAI's embeddings call moved out of app/llm/embeddings.py, the one core module that imports a provider SDK (#33)",
    "backend/tests/test_llm_core_embeddings.py": "new: tests of the core's embeddings and OpenAI's call (#33)",
    "backend/tests/test_llm_langchain_callbacks.py": "new: tests of BudgetGuard driving the Meter (#33)",
    "backend/tests/test_llm_langchain_embeddings.py": "new: characterises GatewayEmbeddings through its public calls before it moves onto the Meter (#33)",
    "backend/src/app/llm/prompts.py": "moved to app/llm/core/prompts.py and app/llm/langchain/prompts.py; the old names warn until 0.8.0 (#33)",
    "backend/src/app/llm/core/prompts.py": "new: the framework-neutral prompt, its hash and loading its file, moved out of app/llm/prompts.py (#33)",
    "backend/src/app/llm/langchain/prompts.py": "new: LangChain's Prompt with config(), load_prompt and prompt_config, moved out of app/llm/prompts.py (#33)",
    "backend/tests/test_llm_core_prompts.py": "new: tests of the neutral prompt (#33); Gateway comes from app/llm/langchain/gateway.py (#33)",
    "backend/src/app/prompts/README.md": "load_prompt comes from app/llm/langchain/prompts.py (#33)",
    "backend/src/app/llm/core/report.py": "new: the spend report moved out of app/llm/report.py, with cli() as its command line (#33)",
    "backend/migrations/env.py": "imports the checkpointer from app/llm/langchain/checkpoints.py (#33)",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py": "imports the checkpointer from app/llm/langchain/checkpoints.py (#33)",
    "backend/src/app/llm/checkpoints.py": "moved to app/llm/langchain/checkpoints.py; the old names warn until 0.8.0 (#33)",
    "backend/src/app/llm/langchain/checkpoints.py": "new: LangGraph's Postgres checkpointer moved out of app/llm/checkpoints.py (#33)",
    "backend/src/app/llm/langchain/gateway.py": "new: the gateway moved out of app/llm/gateway.py, with GatewayDep importable as app.llm.GatewayDep (#33)",
    "backend/tests/test_llm_report_cli.py": "new: pins the spend report's command line, on the new and the old module (#33)",
}

# Lines Copier writes that name where and when the template was applied, not what it generated.
VOLATILE_PREFIXES = ("_commit:", "_src_path:")
ANSWERS_FILE = ".copier-answers.fastapi.yml"


def require_ref(ref: str) -> None:
    """Fail with a clear message when the baseline commit is not in this clone (shallow clone)."""
    found = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=ROOT,
                           capture_output=True)
    if found.returncode != 0:
        pytest.fail(f"BASELINE_REF {ref} is not in this clone. Fetch the full history "
                    "(`git fetch --unshallow`, or `fetch-depth: 0` in actions/checkout).")


def render_at(dst: Path, vcs_ref: str) -> Path:
    """Render with defaults. vcs_ref="HEAD" uses the working tree, uncommitted changes included
    (Copier copies a dirty local template as it is); a SHA renders that commit."""
    dst.mkdir(parents=True)
    (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump(BASE_ANSWERS))
    copier.run_copy(str(ROOT), dst, defaults=True, unsafe=True, quiet=True, vcs_ref=vcs_ref)
    return dst


def read(path: Path) -> bytes:
    data = path.read_bytes()
    if path.name == ANSWERS_FILE:
        lines = data.decode().splitlines(keepends=True)
        data = "".join(ln for ln in lines if not ln.startswith(VOLATILE_PREFIXES)).encode()
    return data


def files_of(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): read(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def describe(path: str, old: bytes | None, new: bytes | None) -> str:
    if old is None:
        return f"  added in HEAD: {path}"
    if new is None:
        return f"  missing in HEAD: {path}"
    try:
        diff = difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(),
                                    "baseline", "HEAD", lineterm="", n=1)
    except UnicodeDecodeError:
        return f"  binary file differs: {path}"
    return "  differs: " + path + "\n" + "\n".join("    " + ln for ln in list(diff)[:20])


def compare(baseline: Mapping[str, bytes], head: Mapping[str, bytes],
            intended: Mapping[str, str]) -> list[str]:
    """Problems found, one entry per path; empty when the renders differ only as intended."""
    differing = {p for p in baseline.keys() | head.keys() if baseline.get(p) != head.get(p)}
    problems = [describe(p, baseline.get(p), head.get(p)) for p in sorted(differing - intended.keys())]
    problems += [f"  INTENDED lists {p} ({intended[p]}) but it no longer differs: remove it"
                 for p in sorted(intended.keys() - differing)]
    return problems


def test_default_render_matches_baseline(tmp_path):
    require_ref(BASELINE_REF)
    baseline = files_of(render_at(tmp_path / "baseline", BASELINE_REF))
    head = files_of(render_at(tmp_path / "head", "HEAD"))
    problems = compare(baseline, head, INTENDED)
    assert not problems, (f"The default render differs from BASELINE_REF {BASELINE_REF[:7]}. "
                          "Add each path to INTENDED with a reason, or fix the change:\n"
                          + "\n".join(problems))


def test_compare_reports_every_unlisted_difference_and_every_stale_entry():
    base = {"a.txt": b"one\n", "b.txt": b"same\n", "gone.txt": b"x\n", "bin": b"\xff\x00"}
    head = {"a.txt": b"two\n", "b.txt": b"same\n", "new.txt": b"y\n", "bin": b"\xfe\x00"}
    problems = compare(base, head, {"b.txt": "was changed (#1)", "new.txt": "added (#2)"})
    text = "\n".join(problems)
    assert "differs: a.txt" in text and "-one" in text and "+two" in text
    assert "missing in HEAD: gone.txt" in text
    assert "binary file differs: bin" in text
    assert "INTENDED lists b.txt" in text
    assert "new.txt" not in text.replace("added (#2)", "")
    assert compare(base, base, {}) == []


def test_answers_file_ignores_only_commit_and_src_path(tmp_path):
    f = tmp_path / ANSWERS_FILE
    f.write_text("_commit: v1\n_src_path: /tmp/x\npython_version: '3.14'\n")
    assert read(f) == b"python_version: '3.14'\n"


def test_a_missing_baseline_ref_fails_with_a_fetch_hint():
    with pytest.raises(pytest.fail.Exception, match="Fetch the full history"):
        require_ref("0" * 40)


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "-c", "commit.gpgsign=false",
         "-c", "tag.gpgSign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout


def tracked(dst: Path) -> set[str]:
    return set(git(dst, "ls-files").splitlines())


def committed_baseline_project(path: Path) -> Path:
    """A project rendered at BASELINE_REF, plus one file of the user's, committed in git."""
    require_ref(BASELINE_REF)
    dst = render_at(path, BASELINE_REF)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend" / "user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    return dst


def update_to_head(dst: Path) -> None:
    copier.run_update(dst, answers_file=ANSWERS_FILE, defaults=True, unsafe=True, overwrite=True,
                      quiet=True, vcs_ref="HEAD")


def use_project_base_answers(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    """Make Copier's temporary "old copy" of the template see the project's base answers.

    FOUND WHILE WRITING THIS TEST (#49): `copier update` renders the old template version into an
    empty temporary directory (copier/_main.py in Copier 9.19.1, `_apply_update`), where
    `_external_data.base` (.copier-answers.base.yml) does not exist. The old copy uses the OLD
    version's own copier.yml, whose python_version validator then fails with "Apply
    house-template-base first". That validator is in every release up to BASELINE_REF, so fixing
    HEAD (#49) does not change this: the workaround and the test below it can only go once
    BASELINE_REF is a release that contains the #49 fix.

    This is a stopgap that patches a private Copier function. It proves "with the base answers
    visible, a defaults update deletes nothing and leaves no conflicts". It does NOT prove that
    users can update (#49).
    """
    real = copier._main.load_answersfile_data

    def load(dst_path: Path, answers_file: Path | str = ".copier-answers.yml", **kw: bool):
        found = real(dst_path, answers_file, **{**kw, "warn_on_missing": False})
        return found or real(project, answers_file, **kw)

    monkeypatch.setattr(copier._main, "load_answersfile_data", load)


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_updating_from_the_baseline_cannot_read_the_base_answers_yet(tmp_path):
    # Known bug (#49), see use_project_base_answers. Fails with "DID NOT RAISE" once BASELINE_REF
    # is a release that contains the fix; then delete this test and the stopgap.
    dst = committed_baseline_project(tmp_path / "project")
    with pytest.raises(ValueError, match="Apply house-template-base first"):
        update_to_head(dst)


def test_updating_a_baseline_project_with_defaults_deletes_nothing(tmp_path, monkeypatch):
    # Proves: `copier update --defaults` from BASELINE_REF to the working tree (vcs_ref="HEAD")
    # keeps every file the project had, keeps a file the user added, leaves no conflict markers or
    # .rej files, and brings the project to what a fresh render at HEAD produces. Copier deletes
    # files that an answer change stops generating (#32); the defaults must not stop generating any.
    # PROVEN here (Copier 9.19.1, #29): questions added after the baseline (`storage`, `database`) take their
    # default under `defaults=True`, and the answers file gets it. The docs do not say so
    # (https://copier.readthedocs.io/en/v9.19.1/updating/). The project's lockfile is its own
    # (`_skip_if_exists`), so it keeps the old content: the user runs `make lock`.
    dst = committed_baseline_project(tmp_path / "project")
    before = tracked(dst)
    baseline_lock = (dst / "backend" / "uv.lock").read_bytes()
    use_project_base_answers(monkeypatch, dst)

    update_to_head(dst)

    after = files_of(dst)
    assert before - after.keys() == set(), "update deleted files"
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert not list(dst.rglob("*.rej"))
    assert [p for p, data in after.items() if b"<<<<<<<" in data] == []
    # The update really ran: the project now equals a fresh render at HEAD (plus the user's file).
    fresh = files_of(render_at(tmp_path / "fresh", "HEAD"))
    after.pop("backend/user_added.txt")
    recorded = yaml.safe_load((dst / ANSWERS_FILE).read_text())
    assert (recorded["storage"], recorded["database"]) == (True, True)
    assert after.pop("backend/uv.lock") == baseline_lock
    fresh.pop("backend/uv.lock")
    assert after == fresh


def test_updating_a_baseline_project_with_storage_off_deletes_only_the_storage_files(
    tmp_path, monkeypatch
):
    # A project made before the `storage` question existed (no such answer in its answers file)
    # turns it off. The NEW template's `_exclude` patterns are also rendered for the old copy of
    # the project, whose answers lack `storage`: `storage is defined` in copier.yml keeps that
    # from failing with StrictUndefined (#29).
    dst = committed_baseline_project(tmp_path / "project")
    assert "storage" not in yaml.safe_load((dst / ANSWERS_FILE).read_text())
    before = tracked(dst)
    use_project_base_answers(monkeypatch, dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data={"storage": False}, defaults=True,
                      unsafe=True, overwrite=True, quiet=True, vcs_ref="HEAD")

    removed = before - set(files_of(dst))
    assert removed == {"backend/src/app/storage.py", "backend/tests/storage.py",
                       "backend/tests/test_storage.py"}
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert not (dst / "backend/compose/storage").exists()


def test_updating_a_baseline_project_with_database_off_deletes_only_the_database_files(
    tmp_path, monkeypatch
):
    # Like the storage test above, for `database`: the baseline project has no such answer, so the
    # `database is defined` guard in copier.yml keeps the rendering of the OLD copy from failing
    # with StrictUndefined (#29). Directory patterns (backend/migrations) must remove everything
    # inside them, and only the template's files: the user's file stays.
    dst = committed_baseline_project(tmp_path / "project")
    assert "database" not in yaml.safe_load((dst / ANSWERS_FILE).read_text())
    before = tracked(dst)
    use_project_base_answers(monkeypatch, dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data={"database": False}, defaults=True,
                      unsafe=True, overwrite=True, quiet=True, vcs_ref="HEAD")

    removed = before - set(files_of(dst))
    assert removed, "nothing was removed"
    assert {"backend/src/app/db.py", "backend/src/app/llm/gateway.py",
            "backend/migrations/env.py"} <= removed
    assert not any(p.startswith(("backend/migrations/", "backend/src/app/llm/")) for p in
                   set(files_of(dst)))
    assert not (dst / "backend/compose/db").exists()
    # Storage and the always-present files stay; so does the user's file.
    assert "backend/src/app/storage.py" not in removed
    assert "backend/src/app/main.py" not in removed
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["database"] is False
