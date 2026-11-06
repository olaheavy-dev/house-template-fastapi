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
from collections.abc import Iterable, Mapping
from pathlib import Path

import copier
import pytest
import yaml

from conftest import BASE_ANSWERS, ROOT

# main at d67ca38 ("no default models or prices", #42), the state #29 must not change by accident.
BASELINE_REF = "d67ca3807ae74a78cc9728bf20da829be035117c"

# {rendered path: "reason (#NN)"}. Starts empty. The test fails for a path that differs without an
# entry here, and for an entry whose path no longer differs.
INTENDED: dict[str, str] = {}

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
    return {p.relative_to(root).as_posix(): read(p) for p in sorted(root.rglob("*")) if p.is_file()}


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
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args], cwd=cwd,
        capture_output=True, text=True, check=True)
    return result.stdout


def tracked(dst: Iterable[str] | Path) -> set[str]:
    return set(git(Path(str(dst)), "ls-files").splitlines())


def use_project_base_answers(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    """Make Copier's temporary "old copy" of the template see the project's base answers.

    FOUND WHILE WRITING THIS TEST: `copier update` renders the old template version into an empty
    temporary directory (copier/_main.py, `_apply_update`), where `_external_data.base`
    (.copier-answers.base.yml) does not exist, so the python_version validator fails with
    "Apply house-template-base first". This patch hides that for the test only. Remove it, and the
    xfail below, when the template handles it.
    """
    real = copier._main.load_answersfile_data

    def load(dst_path: Path, answers_file: Path | str = ".copier-answers.yml", **kw: bool):
        found = real(dst_path, answers_file, **{**kw, "warn_on_missing": False})
        return found or real(project, answers_file, **kw)

    monkeypatch.setattr(copier._main, "load_answersfile_data", load)


@pytest.mark.parametrize("patched", [
    True,
    pytest.param(False, marks=pytest.mark.xfail(
        raises=ValueError, strict=True, reason="copier update can't read the base answers (#29)")),
])
def test_updating_a_baseline_project_with_defaults_deletes_nothing(tmp_path, monkeypatch, patched):
    # Proves: `copier update --defaults` from BASELINE_REF to the working tree (vcs_ref="HEAD")
    # keeps every file the project had, keeps a file the user added, and leaves no conflict
    # markers or .rej files. Copier deletes files that an answer change stops generating (#32);
    # the defaults must not stop generating any.
    # UNVERIFIED in the docs: whether a question added after the baseline takes its default under
    # `defaults=True` (https://copier.readthedocs.io/en/v9.19.1/updating/ does not say). Later PRs
    # of #29 add questions, and this test then starts proving it; it needs no change for that.
    require_ref(BASELINE_REF)
    dst = render_at(tmp_path / "project", BASELINE_REF)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend" / "user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    before = tracked(dst)
    if patched:
        use_project_base_answers(monkeypatch, dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, defaults=True, unsafe=True, overwrite=True,
                      quiet=True, vcs_ref="HEAD")

    existing = {p.relative_to(dst).as_posix() for p in dst.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(dst).parts}
    assert before - existing == set(), "update deleted files"
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert not list(dst.rglob("*.rej"))
    conflicted = [p for p in existing if b"<<<<<<<" in (dst / p).read_bytes()]
    assert conflicted == []
