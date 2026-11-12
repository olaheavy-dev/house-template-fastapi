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
- Dependabot PRs that change only `uv.lock` or an image tag or digest pass; one that edits
  `pyproject.toml` still needs an INTENDED entry. The comparison ignores what Dependabot maintains
  in the rendered output (see DEPENDABOT_LOCK_FILES and DEPENDABOT_COMPOSE_FILE): the generated
  `backend/uv.lock` completely, and the tag or digest of the `image:` lines in the compose files
  under `backend/compose/`. Any other change in those files, and any other file, still counts.
  Dockerfile `FROM` lines and the workflows are not ignored: `.github/dependabot.yml` does not
  maintain them in this repository (a generated project's own Dependabot does).
"""

from __future__ import annotations

import difflib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import copier
import pytest
import yaml

from conftest import BASE_ANSWERS, ROOT

# The v0.9.0 release (tag; commit a700ee9), the state the default render may only leave on purpose.
# It records the `llm` answer, so a baseline project updates to HEAD without the 0.7.x fallbacks.
BASELINE_REF = "v0.9.0"

# The files 0.8.0 removed (#33): the old import paths and the tests that exist only for them. A
# project made from v0.7.x or earlier loses them on `copier update` to this version or later;
# test_update.py checks that from those old projects.
REMOVED_FILES = {
    *(f"backend/src/app/llm/{name}.py" for name in (
        "budget", "checkpoints", "embeddings", "errors", "gateway", "metrics", "prompts", "report")),
    "backend/tests/test_llm_shims.py",
}

# {rendered path: "reason (#NN)"}. Starts empty after each baseline move. The test fails for a path
# that differs without an entry here, and for an entry whose path no longer differs. A path-level
# entry hides a second, accidental change to the same file until the next baseline move.
# Not compared: file modes, symlinks and empty directories (the template has no tracked
# executables today).
INTENDED: dict[str, str] = {
    ".claude/rules/fastapi.md":
        "names the maf extra and the `llm: maf` answer in the extras sentence; tool definitions and "
        "output schemas are redacted; the structured-output 502 (#35)",
    "backend/pyproject.toml":
        "the maf extra with exact pins, [tool.uv] conflicts, the MAF packages in the import contract (#35)",
    "backend/src/app/llm/core/openai_embeddings.py":
        "openai_client takes an optional http_client, a test seam for the MAF gateway's SDK client (#35)",
    "backend/src/app/settings.py":
        "hide_input_in_errors: a validation error never echoes the value given (#35)",
    "backend/tests/test_llm_parity_redaction.py":
        "shared chat redaction tests, run once per framework: tool definitions and schemas, streams, "
        "the structured-output 502, rows told apart by count not created_at (#35)",
    "backend/src/app/llm/core/redaction_data.py":
        "walk_definition and with_schema: tool definitions and output schemas are redacted (#112 review)",
    "backend/src/app/llm/langchain/redaction.py":
        "bound tool definitions and output schemas are redacted before the call (#112 review); a "
        "stream stopped early ends its inner run at once; structured-output parse errors are the "
        "core's 502 (#35 PR 3)",
    "backend/src/app/llm/langchain/errors.py":
        "OutputParserException and StructuredOutputError are the core's 502 (#35 PR 3)",
    "backend/tests/harness/__init__.py":
        "the harness can offer a tool, ask for structured output, stream and run an agent (#35)",
    "backend/migrations/versions/20261112_0007_agent_session.py":
        "migration 0007: the agent_session table, with any model gateway (#35 PR 5)",
    "backend/src/app/llm/core/sessions.py":
        "the stored agent session row and SessionConflict (#35 PR 5)",
    "backend/tests/test_llm_parity_threads.py":
        "an image in a thread across a restart, the same under every framework (#73, #35 PR 5)",
    "backend/src/app/models.py":
        "imports AgentSessionRow, so autogenerate sees the agent_session table (#35 PR 5)",
    ".github/template-owned/fastapi.txt":
        "lists migration 0007 as template-owned (#35 PR 5)",
    "backend/tests/test_llm_migrations.py":
        "the template's chain now ends at 0007 (#35 PR 5)",
    "backend/tests/test_llm_parity_steps.py":
        "the step limit's shared tests, run once per framework (#35 PR 4)",
    "backend/tests/test_llm_steps.py":
        "the framework fixture's note points to the shared step tests (#35 PR 4)",
    "backend/tests/harness/langchain.py":
        "the scripted model binds tools and structured output as ChatOpenAI does, streams, and "
        "loops on a tool for the step-limit tests (#35)",
    "backend/tests/test_llm_core_redaction_definitions.py":
        "the core's rules for redacting tool definitions and output schemas (#112 review)",
    "backend/tests/test_llm_redaction_structured.py":
        "an output schema reaches the wire with placeholders, through every method; a placeholder "
        "a validator rejects is the 502 (#35)",
    "backend/tests/test_llm_redaction_tools.py":
        "a tool's definition reaches the wire with placeholders, also when streamed (#112 review)",
    "backend/llm-contracts.toml":
        "the core contract also forbids the MAF provider packages; the comment is accurate (#35)",
    "backend/src/app/llm/core/errors.py":
        "a bug is logged with the type names of its chain, not a traceback whose messages can hold "
        "provider text; SchemaMismatch, the structured-output 502 (#35)",
    "backend/tests/test_llm_core_errors.py":
        "the log field is the chain of type names; a test pins that no text from the chain is logged (#35)",
    "backend/tests/test_llm_error_logs.py":
        "the log field is the chain of type names (#35)",
    "backend/tests/test_llm_prices.py":
        "built on the neutral Usage, so it runs under every framework (#35)",
    "backend/tests/test_llm_cost_pins.py":
        "built on the neutral Usage, so it runs under every framework (#35)",
    "backend/tests/test_llm_parity_budget.py":
        "a shared test that a long tool definition is reserved for in full; streamed calls (#35)",
}

# DEPENDABOT_*: what Dependabot updates in this repository's rendered output
# (.github/dependabot.yml). Its PRs must pass this test unchanged, so these are normalised away
# before comparing; nothing else is.
# - uv.lock: generated, never hand-edited, so the whole file is skipped.
# - `image:` lines in the compose files: only the tag and digest are masked; the image name, and
#   every other line of the file, still count. The mask drops everything after the first `:` or `@`
#   in the last path segment, so losing the pin or changing the tag by hand passes this test; the
#   pin format is checked by test_render.py (test_local_database_is_pinned_and_keeps_its_data,
#   test_local_storage_is_pinned_private_and_healthchecked) and by test_options.py
#   (test_the_pins_dependabot_reads_live_in_plain_compose_files).
DEPENDABOT_LOCK_FILES = frozenset({"backend/uv.lock"})
DEPENDABOT_COMPOSE_FILE = re.compile(r"backend/compose/[^/]+/compose\.ya?ml")
IMAGE_LINE = re.compile(rb"^(\s*image:\s*)(\S+)(\s*)$")

# Lines Copier writes that name where and when the template was applied, not what it generated.
VOLATILE_PREFIXES = ("_commit:", "_src_path:")
ANSWERS_FILE = ".copier-answers.fastapi.yml"


def require_ref(ref: str) -> None:
    """Fail with a clear message when the baseline commit is not in this clone (shallow clone)."""
    found = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=ROOT,
                           capture_output=True)
    if found.returncode != 0:
        pytest.fail(f"Commit {ref} is not in this clone. Fetch the full history "
                    "(`git fetch --unshallow --tags`, or `fetch-depth: 0` in actions/checkout).")


def render_at(dst: Path, vcs_ref: str, **data: object) -> Path:
    """Render with defaults and any answers given. vcs_ref="HEAD" uses the working tree, uncommitted
    changes included (Copier copies a dirty local template as it is); a tag or SHA renders that
    commit."""
    dst.mkdir(parents=True)
    (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump(BASE_ANSWERS))
    copier.run_copy(str(ROOT), dst, data=data, defaults=True, unsafe=True, quiet=True,
                    vcs_ref=vcs_ref)
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


def mask_image_version(line: bytes) -> bytes:
    """`image: repo/name:tag@sha256:...` becomes `image: repo/name:<pinned>`; other lines are kept."""
    match = IMAGE_LINE.match(line)
    if not match:
        return line
    prefix, reference, suffix = match.groups()
    registry_and_path, slash, last = reference.rpartition(b"/")
    name = re.split(rb"[:@]", last, maxsplit=1)[0]
    return prefix + registry_and_path + slash + name + b":<pinned>" + suffix


def ignore_dependabot_changes(files: Mapping[str, bytes]) -> dict[str, bytes]:
    """The files without what Dependabot maintains (the constants above)."""
    result = {}
    for path, data in files.items():
        if path in DEPENDABOT_LOCK_FILES:
            continue
        if DEPENDABOT_COMPOSE_FILE.fullmatch(path):
            data = b"\n".join(mask_image_version(ln) for ln in data.split(b"\n"))
        result[path] = data
    return result


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
    baseline = ignore_dependabot_changes(baseline)
    head = ignore_dependabot_changes(head)
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


def test_database_off_render_matches_baseline(tmp_path):
    # Without the database there is no gateway, and the answers file records `llm: none`. Only the
    # files every render shares differ on purpose (a subset of INTENDED).
    require_ref(BASELINE_REF)
    baseline = files_of(render_at(tmp_path / "baseline", BASELINE_REF, database=False))
    head = files_of(render_at(tmp_path / "head", "HEAD", database=False))
    shared = {path: INTENDED[path] for path in (".claude/rules/fastapi.md", "backend/pyproject.toml",
                                                "backend/src/app/settings.py")}
    problems = compare(baseline, head, shared)
    assert not problems, (f"The database-off render differs from BASELINE_REF {BASELINE_REF[:7]}:\n"
                          + "\n".join(problems))
    assert b"llm: none\n" in head[ANSWERS_FILE]


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


COMPOSE = "backend/compose/db/compose.yaml"
COMPOSE_TEXT = (b"services:\n  db:\n    # pinned (Dependabot updates it).\n"
                b"    image: pgvector/pgvector:0.8.7-pg18-trixie@sha256:aaa\n"
                b"    environment:\n      POSTGRES_USER: app\n")


def test_compare_ignores_a_change_to_the_lock_file_only():
    base = {"backend/uv.lock": b"vcrpy 8.1.0\n", "a.txt": b"x\n"}
    head = {"backend/uv.lock": b"vcrpy 8.2.0\n", "a.txt": b"x\n"}
    assert compare(base, head, {}) == []


def test_compare_ignores_an_image_tag_or_digest_change_in_a_compose_file():
    new = COMPOSE_TEXT.replace(b"0.8.7-pg18-trixie@sha256:aaa", b"0.9.0-pg18-trixie@sha256:bbb")
    assert new != COMPOSE_TEXT
    assert compare({COMPOSE: COMPOSE_TEXT}, {COMPOSE: new}, {}) == []
    digest_only = COMPOSE_TEXT.replace(b"sha256:aaa", b"sha256:bbb")
    assert compare({COMPOSE: COMPOSE_TEXT}, {COMPOSE: digest_only}, {}) == []


def test_compare_still_reports_other_changes_in_a_compose_file():
    other = COMPOSE_TEXT.replace(b"POSTGRES_USER: app", b"POSTGRES_USER: root")
    assert "POSTGRES_USER: root" in "\n".join(compare({COMPOSE: COMPOSE_TEXT}, {COMPOSE: other}, {}))
    # A different image name is not a version change.
    renamed = COMPOSE_TEXT.replace(b"pgvector/pgvector:", b"other/image:")
    assert "differs: " + COMPOSE in "\n".join(compare({COMPOSE: COMPOSE_TEXT}, {COMPOSE: renamed}, {}))
    # A comment change is not an image line.
    commented = COMPOSE_TEXT.replace(b"pinned", b"unpinned")
    assert compare({COMPOSE: COMPOSE_TEXT}, {COMPOSE: commented}, {}) != []
    # The mask is for `image:` only: not another key, and not a commented-out image line.
    for old, new in ((b"    command: a:1\n", b"    command: a:2\n"),
                     (b"    # image: a:1\n", b"    # image: a:2\n")):
        assert compare({COMPOSE: old}, {COMPOSE: new}, {}) != [], old
    # A registry with a port keeps its host in the name.
    assert mask_image_version(b"    image: localhost:5000/app:1") == (
        b"    image: localhost:5000/app:<pinned>")


def test_compare_still_reports_the_same_change_in_other_files():
    for path in ("backend/Dockerfile", "backend/pyproject.toml", ".github/workflows/ci.yaml",
                 "backend/compose.yaml", "backend/compose/db/other.txt"):
        for old, new in ((b"    image: a:1\n", b"    image: a:2\n"), (b"uses: x@v1\n", b"uses: x@v2\n")):
            problems = compare({path: old}, {path: new}, {})
            assert len(problems) == 1 and path in problems[0], path


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


def committed_baseline_project(path: Path, ref: str = BASELINE_REF, **data: object) -> Path:
    """A project rendered at BASELINE_REF (or another ref), plus one file of the user's, committed
    in git."""
    require_ref(ref)
    dst = render_at(path, ref, **data)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend" / "user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    return dst


def update_to_head(dst: Path) -> None:
    copier.run_update(dst, answers_file=ANSWERS_FILE, defaults=True, unsafe=True, overwrite=True,
                      quiet=True, vcs_ref="HEAD")


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_updating_a_baseline_project_with_defaults_deletes_nothing(tmp_path):
    # Proves: `copier update --defaults` from BASELINE_REF to the working tree (vcs_ref="HEAD")
    # keeps every file the project had, keeps a file the user added, leaves no conflict markers or
    # .rej files, and brings the project to what a fresh render at HEAD produces. Copier deletes
    # files that an answer change stops generating (#32); the defaults must not stop generating any.
    # The baseline (v0.9.0) has the #49 fix: the update works without the base answers file
    # in Copier's temporary old copy (Copier 9.19.1 warns that the file is missing; that is harmless).
    # The project's lockfile is its own (`_skip_if_exists`), so it keeps the old content: the user
    # runs `make lock`.
    dst = committed_baseline_project(tmp_path / "project")
    before = tracked(dst)
    baseline_lock = (dst / "backend" / "uv.lock").read_bytes()

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
    assert (recorded["storage"], recorded["database"], recorded["container_images"]) == (
        True, True, True)
    assert after.pop("backend/uv.lock") == baseline_lock
    fresh.pop("backend/uv.lock")
    assert after == fresh


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_updating_a_baseline_project_with_storage_off_deletes_only_the_storage_files(
    tmp_path
):
    # A baseline project turns `storage` off. The `storage is defined` guard in copier.yml (#29) is
    # for projects made from main after #53 and before this question (#54); v0.6.1 and earlier
    # recopy instead. That range is tested with PRE_OPTIONS in test_update.py; v0.9.0 records the
    # answer.
    dst = committed_baseline_project(tmp_path / "project")
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["storage"] is True
    before = tracked(dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data={"storage": False}, defaults=True,
                      unsafe=True, overwrite=True, quiet=True, vcs_ref="HEAD")

    removed = before - set(files_of(dst))
    assert removed == {"backend/src/app/storage.py", "backend/tests/storage.py",
                       "backend/tests/test_storage.py", "backend/compose/storage/compose.yaml"}
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert not (dst / "backend/compose/storage").exists()


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_updating_a_baseline_project_with_database_off_deletes_only_the_database_files(
    tmp_path
):
    # Like the storage test above, for `database`. Directory patterns (backend/migrations) must
    # remove everything inside them, and only the template's files: the user's file stays.
    dst = committed_baseline_project(tmp_path / "project")
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["database"] is True
    before = tracked(dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data={"database": False}, defaults=True,
                      unsafe=True, overwrite=True, quiet=True, vcs_ref="HEAD")

    removed = before - set(files_of(dst))
    assert removed, "nothing was removed"
    assert {"backend/src/app/db.py", "backend/src/app/llm/langchain/gateway.py",
            "backend/migrations/env.py"} <= removed
    assert not any(p.startswith(("backend/migrations/", "backend/src/app/llm/")) for p in
                   set(files_of(dst)))
    assert not (dst / "backend/compose/db").exists()
    # Storage and the always-present files stay; so does the user's file.
    assert "backend/src/app/storage.py" not in removed
    assert "backend/src/app/main.py" not in removed
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["database"] is False
