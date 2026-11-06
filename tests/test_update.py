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
import yaml

from conftest import BASE_ANSWERS, ROOT
from test_golden_render import ANSWERS_FILE, files_of, git, require_ref, tracked

PUBLISH = ".github/workflows/publish-api.yml"
SKIP = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".mypy_cache", ".ruff_cache",
                              ".pytest_cache", ".import_linter_cache", ".env", ".env.*")


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
    rules = repo / "template/.claude/rules/fastapi.md"
    rules.write_text(rules.read_text() + "\nA rule added by the second version.\n")
    publish = next((repo / "template/.github/workflows").glob("*publish-api.yml*.jinja"))
    publish.write_text(publish.read_text() + "# A line added by the second version.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: v100.0.1")
    git(repo, "tag", "v100.0.1")
    return repo


def copy_layer(src: Path | str, dst: Path, ref: str, **data: object) -> None:
    copier.run_copy(str(src), dst, data=data, defaults=True, unsafe=True, quiet=True, vcs_ref=ref)


def update(dst: Path, ref: str) -> None:
    copier.run_update(dst, answers_file=ANSWERS_FILE, defaults=True, unsafe=True, overwrite=True,
                      quiet=True, vcs_ref=ref)


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
    # The two renders agree on publish-api.yml and on the template-owned list, for both answers.
    assert after.get(PUBLISH) == expected.get(PUBLISH)
    assert (PUBLISH in after) is images
    owned = ".github/template-owned/fastapi.txt"
    assert after[owned] == expected[owned]
    if images:
        assert b"A line added by the second version." in after[PUBLISH]


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_copy_task_refuses_without_base_answers(tmp_path):
    dst = tmp_path / "project"
    dst.mkdir()
    with pytest.raises(Exception) as caught:  # noqa: B017 (the task's failure type is Copier's)
        copy_layer(ROOT, dst, "HEAD", container_images=True)
    text = str(caught.value) + str(getattr(caught.value, "stderr", "")) + str(
        getattr(caught.value, "stdout", ""))
    assert "Apply house-template-base first" in text
    assert "git clean" in text


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
def test_recopy_moves_a_v0_6_1_project_to_head(tmp_path):
    require_ref("v0.6.1")
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(ROOT, dst, "v0.6.1")
    assert "container_images" not in answers(dst)
    git(dst, "init", "-q", "-b", "main")
    lock = dst / "backend/uv.lock"
    lock.write_text(lock.read_text() + "# my lock edit\n")
    (dst / "backend/src/app/health.py").write_text("# my edit to a template-owned file\n")
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    before = tracked(dst)

    copier.run_recopy(dst, answers_file=ANSWERS_FILE, defaults=True, overwrite=True, unsafe=True,
                      quiet=True, vcs_ref="HEAD")

    assert answers(dst)["container_images"] is True
    assert lock.read_text().endswith("# my lock edit\n")
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(ROOT, fresh, "HEAD")
    assert (dst / "backend/src/app/health.py").read_bytes() == (
        fresh / "backend/src/app/health.py").read_bytes()
    assert before - {p for p in files_of(dst)} == set(), "recopy deleted files"
    # After the one-time recopy, normal updates work (the project is committed first).
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: recopy")
