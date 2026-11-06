"""Updating a project that was made from this layer (#49).

Docs: Copier 9.19.1
- `_tasks` with `when` and `_copier_operation`: https://copier.readthedocs.io/en/v9.19.1/configuring/
- Updating and `recopy`: https://copier.readthedocs.io/en/v9.19.1/updating/

`copier update` renders the old template version into an empty temporary directory, so nothing a
template reads from .copier-answers.base.yml may decide what it generates: the layer records the
one value it needs (`container_images`) as its own answer. These tests use a throwaway git repository
of this working tree, tagged twice, so they update between two versions that both contain the fix.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import copier
import pytest
from copier.errors import TaskError, UnsafeTemplateError
import yaml

from conftest import BASE_ANSWERS, ROOT
from test_golden_render import ANSWERS_FILE, files_of, git, require_ref, tracked

PUBLISH = ".github/workflows/publish-api.yml"
_CACHES = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache",
                                 ".pytest_cache", ".import_linter_cache")


def SKIP(directory: str, names: list[str]) -> set[str]:  # noqa: N802
    """Skip caches and a developer's .env files, but keep .env.example (as copier.yml's _exclude)."""
    return set(_CACHES(directory, names)) | {
        n for n in names if (n == ".env" or n.startswith(".env."))
        and n not in {".env.example", ".env.example.jinja"}}


def write_base(dst: Path, **overrides: object) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump({**BASE_ANSWERS, **overrides}))


@pytest.fixture
def tagged_repo(tmp_path: Path) -> Path:
    """This working tree as a git repository tagged v100.0.0, then v100.0.1 with a small change."""
    repo = tmp_path / "template-repo"
    shutil.copytree(ROOT, repo, ignore=SKIP)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: v100.0.0")
    git(repo, "tag", "v100.0.0")
    rules = repo / "template/.claude/rules/fastapi.md.jinja"
    rules.write_text(rules.read_text() + "\nA rule added by the second version.\n")
    publish = next((repo / "template/.github/workflows").glob("*publish-api.yml*.jinja"))
    publish.write_text(publish.read_text() + "# A line added by the second version.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: v100.0.1")
    git(repo, "tag", "v100.0.1")
    return repo


def copy_layer(src: Path | str, dst: Path, ref: str, **data: object) -> None:
    copier.run_copy(str(src), dst, data=data, defaults=True, unsafe=True, quiet=True, vcs_ref=ref)


def update(dst: Path, ref: str, unsafe: bool = True, **data: object) -> None:
    copier.run_update(dst, answers_file=ANSWERS_FILE, data=data, defaults=True, unsafe=unsafe,
                      overwrite=True, quiet=True, vcs_ref=ref)


def committed_project(repo: Path, dst: Path, ref: str, **base: object) -> Path:
    write_base(dst, **base)
    copy_layer(repo, dst, ref)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend" / "user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    return dst


def answers(dst: Path) -> dict[str, object]:
    return yaml.safe_load((dst / ANSWERS_FILE).read_text())


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
@pytest.mark.parametrize("images", [True, False])
def test_update_between_two_fixed_versions_needs_no_patch(tmp_path, tagged_repo, images):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0", container_images=images)
    (dst / "backend/src/app/health.py").write_text("# my own edit\n")
    git(dst, "commit", "-q", "-am", "feat: my edit")

    update(dst, "v100.0.1")

    assert answers(dst)["_commit"] == "v100.0.1"
    assert "A rule added by the second version." in (dst / ".claude/rules/fastapi.md").read_text()
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert (dst / "backend/src/app/health.py").read_text() == "# my own edit\n"
    assert not list(dst.rglob("*.rej")) and not list(dst.rglob("*.orig"))
    after = files_of(dst)
    assert [p for p, data in after.items() if b"<<<<<<<" in data] == []
    fresh = tmp_path / "fresh"
    write_base(fresh, container_images=images)
    copy_layer(tagged_repo, fresh, "v100.0.1")
    expected = files_of(fresh)
    # The whole tree equals a fresh render at v100.0.1, apart from the user's file and edit.
    after.pop("backend/user_added.txt")
    assert after.pop("backend/src/app/health.py") == b"# my own edit\n"
    expected.pop("backend/src/app/health.py")
    assert after == expected
    assert (PUBLISH in after) is images
    if images:
        assert b"A line added by the second version." in after[PUBLISH]


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_update_needs_trust_because_the_template_has_tasks(tmp_path, tagged_repo):
    # Copier 9.19.1 `_check_unsafe` counts any `_tasks`, even one whose `when` is false, as unsafe
    # (`copier update --help`: --trust allows "Jinja extensions, migrations, tasks").
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    with pytest.raises(UnsafeTemplateError, match="tasks"):
        update(dst, "v100.0.1", unsafe=False)
    update(dst, "v100.0.1")
    assert answers(dst)["_commit"] == "v100.0.1"


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_update_cannot_check_for_the_base_answers(tmp_path, tagged_repo):
    # Only copy and recopy refuse without the base answers; an update just carries on (README).
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    (dst / ".copier-answers.base.yml").unlink()
    git(dst, "commit", "-q", "-am", "chore: lose the base answers")
    update(dst, "v100.0.1")
    assert answers(dst)["_commit"] == "v100.0.1"


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_copy_task_refuses_without_base_answers(tmp_path):
    dst = tmp_path / "project"
    dst.mkdir()
    with pytest.raises(TaskError) as caught:  # the command text carries the message
        copy_layer(ROOT, dst, "HEAD", container_images=True)
    assert "Apply house-template-base first" in str(caught.value)
    assert "no uncommitted work" in str(caught.value)
    assert "git clean -fd" in str(caught.value)
    assert "delete the directory" not in str(caught.value)


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_copy_without_base_answers_and_no_value_is_refused_before_writing(tmp_path):
    dst = tmp_path / "project"
    dst.mkdir()
    with pytest.raises(ValueError, match="container_images"):
        copy_layer(ROOT, dst, "HEAD")
    assert list(dst.iterdir()) == []


@pytest.mark.parametrize("images", [True, False])
def test_stack_check_still_refuses_without_fastapi(tmp_path, images):
    dst = tmp_path / "project"
    write_base(dst, stacks=["nextjs"], container_images=images)
    with pytest.raises(ValueError, match="don't list the fastapi stack"):
        copy_layer(ROOT, dst, "HEAD")


def test_a_container_images_value_that_disagrees_with_the_base_is_refused(tmp_path):
    dst = tmp_path / "project"
    write_base(dst, container_images=True)
    with pytest.raises(ValueError, match="container_images"):
        copy_layer(ROOT, dst, "HEAD", container_images=False)


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_container_images_follows_a_changed_base_answer_on_update(tmp_path, tagged_repo):
    # Guards undocumented Copier behaviour (a validator that fails on the last answer makes Copier
    # drop it and recompute the default). Re-run on every Copier bump (#49).
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0", container_images=True)
    assert (dst / PUBLISH).is_file()
    write_base(dst, container_images=False)
    git(dst, "commit", "-q", "-am", "chore: base turned images off")

    update(dst, "v100.0.1")

    assert answers(dst)["container_images"] is False
    assert not (dst / PUBLISH).exists()
    assert PUBLISH not in (dst / ".github/template-owned/fastapi.txt").read_text()
    assert (dst / "backend/user_added.txt").exists()


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_container_images_turning_on_adds_the_publish_workflow_on_update(tmp_path, tagged_repo):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0", container_images=False)
    assert not (dst / PUBLISH).exists()
    write_base(dst, container_images=True)
    git(dst, "commit", "-q", "-am", "chore: base turned images on")

    update(dst, "v100.0.1")

    assert answers(dst)["container_images"] is True
    assert (dst / PUBLISH).is_file()
    assert PUBLISH in (dst / ".github/template-owned/fastapi.txt").read_text()


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_recopy_moves_a_v0_6_1_project_to_head(tmp_path, tagged_repo):
    """Recopy a v0.6.1 project to a tag of the throwaway repo, then update to the next tag."""
    require_ref("v0.6.1")
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(ROOT, dst, "v0.6.1")
    assert "container_images" not in answers(dst)
    # Recopy and update use the recorded source, so point it at the repo that has later tags.
    recorded = (dst / ANSWERS_FILE).read_text()
    (dst / ANSWERS_FILE).write_text(
        "".join(f"_src_path: {tagged_repo}\n" if ln.startswith("_src_path:") else ln
                for ln in recorded.splitlines(keepends=True)))
    git(dst, "init", "-q", "-b", "main")
    lock = dst / "backend/uv.lock"
    lock.write_text(lock.read_text() + "# my lock edit\n")
    (dst / "backend/src/app/health.py").write_text("# my edit to a template-owned file\n")
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    before = tracked(dst)

    copier.run_recopy(dst, answers_file=ANSWERS_FILE, defaults=True, overwrite=True, unsafe=True,
                      quiet=True, vcs_ref="v100.0.0")

    assert answers(dst)["container_images"] is True
    assert lock.read_text().endswith("# my lock edit\n")
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.0")
    assert (dst / "backend/src/app/health.py").read_bytes() == (
        fresh / "backend/src/app/health.py").read_bytes()
    assert before - set(files_of(dst)) == set(), "recopy deleted files"

    # After the one-time recopy and a commit, a normal update works.
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: recopy")
    update(dst, "v100.0.1")
    assert answers(dst)["_commit"] == "v100.0.1"
    assert "A rule added by the second version." in (dst / ".claude/rules/fastapi.md").read_text()
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    assert not list(dst.rglob("*.rej"))
    assert [p for p, data in files_of(dst).items() if b"<<<<<<<" in data] == []


STORAGE_PATHS = ["backend/src/app/storage.py", "backend/tests/storage.py",
                 "backend/tests/test_storage.py", "backend/compose/storage/compose.yaml"]


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_a_part_off_on_update_deletes_its_files_and_keeps_user_files(tmp_path, tagged_repo):
    # `copier update --trust --defaults -d storage=false` (#29). Copier 9.19.1 deletes the files
    # an answer stops generating, as #32 found; the user's own files stay, and so does the
    # lockfile, which is the project's (the user runs `make lock`).
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    assert all((dst / p).is_file() for p in STORAGE_PATHS)
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: product")
    lock = (dst / "backend/uv.lock").read_bytes()

    update(dst, "v100.0.1", storage=False)

    assert answers(dst)["storage"] is False
    assert [p for p in STORAGE_PATHS if (dst / p).exists()] == []
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    assert (dst / "backend/uv.lock").read_bytes() == lock
    assert not list(dst.rglob("*.rej")) and not list(dst.rglob("*.orig"))
    after = files_of(dst)
    assert [p for p, data in after.items() if b"<<<<<<<" in data] == []
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.1", storage=False)
    expected = files_of(fresh)
    for own in ("backend/user_added.txt", "backend/src/app/product.py", "backend/uv.lock"):
        after.pop(own)
    expected.pop("backend/uv.lock")
    # The whole tree equals a fresh storage=false render, apart from the user's files.
    assert after == expected

    # And back on: the files return.
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: storage off")
    update(dst, "v100.0.1", storage=True)
    assert answers(dst)["storage"] is True
    assert all((dst / p).is_file() for p in STORAGE_PATHS)
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_storage_off_deletes_the_storage_files_even_if_the_user_edited_them(
    tmp_path, tagged_repo
):
    # The README and the rules file warn: commit first, edits to the template's storage files are
    # deleted with them (recover them from git). Proven here (Copier 9.19.1, #29).
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    storage = dst / "backend/src/app/storage.py"
    storage.write_text(storage.read_text() + "\n# my own change\n")
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: edit storage")

    update(dst, "v100.0.1", storage=False)

    assert not storage.exists()
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    assert "my own change" in git(dst, "show", "HEAD:backend/src/app/storage.py")  # recoverable


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_a_project_that_already_has_storage_off_does_not_get_it_back_on_update(
    tmp_path, tagged_repo
):
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(tagged_repo, dst, "v100.0.0", storage=False)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend/user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    assert not any((dst / p).exists() for p in STORAGE_PATHS)

    update(dst, "v100.0.1")

    assert answers(dst)["storage"] is False
    assert not any((dst / p).exists() for p in STORAGE_PATHS)
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert "A rule added by the second version." in (dst / ".claude/rules/fastapi.md").read_text()
