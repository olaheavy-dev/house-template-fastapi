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
from test_golden_render import (ANSWERS_FILE, committed_baseline_project, files_of, git, require_ref,
                                render_at, tracked, update_to_head, REMOVED_FILES)

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
    # Goes when the README stops supporting a recopy from v0.6.1.
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


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_recopy_overwrites_a_v0_6_1_price_table_and_models_but_git_keeps_the_edits(
        tmp_path, tagged_repo):
    """A recopy overwrites every file the template renders: the price table and app/models.py too
    (#33). It leaves no conflict, so the project's edits come back only from `git diff` or
    `git show HEAD:<path>`. Only backend/uv.lock is kept (copier.yml _skip_if_exists)."""
    # Goes when the README stops supporting a recopy from v0.6.1.
    require_ref("v0.6.1")
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(ROOT, dst, "v0.6.1")
    recorded = (dst / ANSWERS_FILE).read_text()
    (dst / ANSWERS_FILE).write_text(
        "".join(f"_src_path: {tagged_repo}\n" if ln.startswith("_src_path:") else ln
                for ln in recorded.splitlines(keepends=True)))
    prices = dst / "backend/src/app/llm/prices.py"
    models = dst / "backend/src/app/models.py"
    entry = '    "openai:gpt-demo": Price(_rates(1, 4, 0.10, 1.25, 2)),\n'
    old = prices.read_text()
    prices.write_text(old.replace("PRICES: dict[str, Price] = {\n", f"PRICES: dict[str, Price] = {{\n{entry}", 1))
    models.write_text(models.read_text() + "from app.course import Course  # noqa: E402,F401\n")
    assert entry in prices.read_text()
    git(dst, "init", "-q", "-b", "main")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: v0.6.1 with a price and a model")
    edited = {p: (dst / p).read_text() for p in ("backend/src/app/llm/prices.py",
                                                 "backend/src/app/models.py")}

    copier.run_recopy(dst, answers_file=ANSWERS_FILE, defaults=True, overwrite=True, unsafe=True,
                      quiet=True, vcs_ref="v100.0.0")

    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.0")
    for rel, text in edited.items():
        assert (dst / rel).read_bytes() == (fresh / rel).read_bytes(), rel  # overwritten
        assert (dst / rel).read_text() != text, rel
        assert git(dst, "show", f"HEAD:{rel}") == text, rel  # recoverable from git
    assert "gpt-demo" not in prices.read_text()
    assert "app.course" not in models.read_text()


SHIMS = ("gateway", "checkpoints", "errors", "embeddings")


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_update_from_0_7_0_removes_the_old_import_paths_and_keeps_the_projects_prices(tmp_path):
    # What 0.8.0 does to a v0.7.0 project (#33): the old modules go, an edited one included, and the
    # price table keeps the project's entry. Observed on Copier 9.19.1; the README says the same.
    dst = committed_baseline_project(tmp_path / "project", "v0.7.0")
    llm = dst / "backend/src/app/llm"
    entry = ('    "openai:gpt-demo": Price(\n        _rates(1, 4, 0.10, 1.25, 2), checked="2026-11-06",\n'
             '        source=OPENAI_PRICING,\n    ),\n')
    prices = (llm / "prices.py").read_text()
    assert "PRICES: dict[str, Price] = {\n" in prices
    (llm / "prices.py").write_text(prices.replace("PRICES: dict[str, Price] = {\n",
                                                  f"PRICES: dict[str, Price] = {{\n{entry}", 1))
    for name in SHIMS:
        with (llm / f"{name}.py").open("a") as f:
            f.write(f"# my edit to {name}\n")
    product = dst / "backend/src/app/api/ask.py"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_text("from app.llm.langchain.gateway import GatewayDep\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: v0.7.0 project with edits")
    product_before = product.read_bytes()

    update_to_head(dst)

    merged = (llm / "prices.py").read_text()
    assert entry in merged  # the project's price entry survived
    assert "_MOVED" not in merged  # and the forwards for the moved names are gone
    assert "<<<<<<<" not in merged
    old = [f"backend/src/app/llm/{n}.py" for n in (*SHIMS, "budget", "metrics", "prompts", "report")]
    assert [p for p in old if (dst / p).exists()] == []
    assert not list(dst.rglob("*.rej")) and not list(dst.rglob("*.orig"))
    assert [p for p, data in files_of(dst).items() if b"<<<<<<<" in data] == []
    assert product.read_bytes() == product_before


# main right after #53 (`copier update` works) and before #54 (`storage`) and #60 (`database`).
PRE_OPTIONS = "91943a4f696d414abb721066926bf1b926cbaad8"
# The files `database: false` removes from a PRE_OPTIONS project (observed on Copier 9.19.1). The
# gateway has no core/ or langchain/ yet and there is no compose/db: those came later.
PRE_OPTIONS_DATABASE_FILES = {
    "backend/migrations/env.py",
    "backend/migrations/script.py.mako",
    "backend/migrations/versions/20261025_0001_enable_pgvector.py",
    "backend/migrations/versions/20261026_0002_llm_budget.py",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py",
    "backend/migrations/versions/20261026_0004_llm_call_details.py",
    "backend/migrations/versions/20261026_0005_per_user_llm_budget.py",
    "backend/migrations/versions/20261026_app_project_root.py",
    "backend/src/app/db.py",
    "backend/src/app/llm/__init__.py",
    "backend/src/app/llm/budget.py",
    "backend/src/app/llm/checkpoints.py",
    "backend/src/app/llm/embeddings.py",
    "backend/src/app/llm/errors.py",
    "backend/src/app/llm/gateway.py",
    "backend/src/app/llm/metrics.py",
    "backend/src/app/llm/prices.py",
    "backend/src/app/llm/prompts.py",
    "backend/src/app/llm/report.py",
    "backend/src/app/models.py",
    "backend/src/app/prompts/README.md",
    "backend/tests/cassettes/test_llm_gateway/test_a_replayed_call_is_answered_and_its_cost_settled.yaml",
    "backend/tests/db.py",
    "backend/tests/evals/__init__.py",
    "backend/tests/evals/conftest.py",
    "backend/tests/evals/test_gateway_evals.py",
    "backend/tests/models.py",
    "backend/tests/recording.py",
    "backend/tests/test_db.py",
    "backend/tests/test_llm_agents.py",
    "backend/tests/test_llm_budget.py",
    "backend/tests/test_llm_embeddings.py",
    "backend/tests/test_llm_gateway.py",
    "backend/tests/test_llm_prices.py",
    "backend/tests/test_observability.py",
    "backend/tests/test_recording_keys.py",
}
PRE_OPTIONS_STORAGE_FILES = {"backend/src/app/storage.py", "backend/tests/storage.py",
                             "backend/tests/test_storage.py"}


def pre_options_project(path: Path) -> Path:
    require_ref(PRE_OPTIONS)
    write_base(path)
    copy_layer(ROOT, path, PRE_OPTIONS)
    assert not {"storage", "database"} & answers(path).keys()
    git(path, "init", "-q", "-b", "main")
    (path / "backend" / "user_added.txt").write_text("mine\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "chore: start")
    return path


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_update_from_before_the_storage_and_database_questions_keeps_every_file(tmp_path):
    # Goes when the README stops supporting updates from main between #53 and #54/#60. README:
    # "Projects made before the `storage` question existed take `storage: true` on update".
    dst = pre_options_project(tmp_path / "project")
    before = tracked(dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, defaults=True, unsafe=True, overwrite=True,
                      quiet=True, vcs_ref="HEAD")

    recorded = answers(dst)
    assert (recorded["storage"], recorded["database"]) == (True, True)
    # Only the old import paths that 0.8.0 removes (they were real modules in this old copy).
    assert before - set(files_of(dst)) == before & REMOVED_FILES, "update deleted other files"
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert not list(dst.rglob("*.rej"))


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
@pytest.mark.parametrize(("answer", "expected"), [
    ("storage", PRE_OPTIONS_STORAGE_FILES), ("database", PRE_OPTIONS_DATABASE_FILES)])
def test_update_from_before_the_questions_with_one_turned_off_removes_only_its_files(
        tmp_path, answer, expected):
    # Goes when the README stops supporting updates from main between #53 and #54/#60. This is
    # what the `is defined` guards in copier.yml are for: the old copy has no such answer.
    dst = pre_options_project(tmp_path / "project")
    before = tracked(dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data={answer: False}, defaults=True,
                      unsafe=True, overwrite=True, quiet=True, vcs_ref="HEAD")

    assert before - set(files_of(dst)) == expected | before & REMOVED_FILES
    assert answers(dst)[answer] is False
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert not list(dst.rglob("*.rej"))


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


DATABASE_PATHS = [
    "backend/src/app/db.py", "backend/src/app/models.py", "backend/src/app/llm/langchain/gateway.py",
    "backend/src/app/prompts/README.md", "backend/migrations/env.py",
    "backend/migrations/versions/20261026_app_project_root.py", "backend/compose/db/compose.yaml",
    "backend/tests/db.py", "backend/tests/test_db.py", "backend/tests/evals/conftest.py",
]
USER_MIGRATION = "backend/migrations/versions/20270101_mine.py"


def assert_clean_tree(dst: Path) -> None:
    assert not list(dst.rglob("*.rej")) and not list(dst.rglob("*.orig"))
    assert [p for p, data in files_of(dst).items() if b"<<<<<<<" in data] == []


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_database_off_on_update_deletes_its_files_and_keeps_user_files(
    tmp_path, tagged_repo
):
    # `copier update --trust --defaults -d database=false` (#29). Directory patterns in `_exclude`
    # (backend/migrations, backend/src/app/llm) remove the template's files inside them, and the
    # user's own file inside a removed directory stays (U5, not stated in the Copier docs).
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    assert all((dst / p).is_file() for p in DATABASE_PATHS)
    (dst / "backend/migrations/versions/20270101_mine.py").write_text("# my migration\n")
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: product")
    lock = (dst / "backend/uv.lock").read_bytes()

    update(dst, "v100.0.1", database=False)

    # Only the database was answered: the gateway, which needs it, goes with it (#30). A last
    # answer that fails validation is dropped and recomputed from the default (undocumented in
    # Copier 9.19.1, so this assertion guards it).
    assert (answers(dst)["database"], answers(dst)["llm"]) == (False, "none")
    assert [p for p in DATABASE_PATHS if (dst / p).exists()] == []
    assert (dst / USER_MIGRATION).read_text() == "# my migration\n"
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert (dst / "backend/uv.lock").read_bytes() == lock
    assert_clean_tree(dst)
    after = files_of(dst)
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.1", database=False)
    expected = files_of(fresh)
    for own in ("backend/user_added.txt", "backend/src/app/product.py", "backend/uv.lock",
                USER_MIGRATION):
        after.pop(own)
    expected.pop("backend/uv.lock")
    # The whole tree equals a fresh database=false render, apart from the user's files.
    assert after == expected


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_database_off_deletes_its_files_even_if_the_user_edited_them(
    tmp_path, tagged_repo
):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    for rel in ("backend/src/app/db.py", "backend/migrations/env.py"):
        path = dst / rel
        path.write_text(path.read_text() + "\n# my own change\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: edit the database files")

    update(dst, "v100.0.1", database=False)

    assert not (dst / "backend/src/app/db.py").exists()
    assert not (dst / "backend/migrations/env.py").exists()
    assert "my own change" in git(dst, "show", "HEAD:backend/src/app/db.py")  # recoverable
    assert "my own change" in git(dst, "show", "HEAD:backend/migrations/env.py")


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_database_back_on_restores_the_template_versions_not_the_edits(
    tmp_path, tagged_repo
):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    db_py = dst / "backend/src/app/db.py"
    original = db_py.read_bytes()
    db_py.write_text(db_py.read_text() + "\n# my own change\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: edit db")
    update(dst, "v100.0.1", database=False)
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: database off")

    # database=true alone no longer brings the gateway back (#30): `llm` stays none.
    update(dst, "v100.0.1", database=True)
    assert (answers(dst)["database"], answers(dst)["llm"]) == (True, "none")
    assert not (dst / "backend/src/app/llm").exists()
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: database on")

    update(dst, "v100.0.1", llm="langchain")

    assert (answers(dst)["database"], answers(dst)["llm"]) == (True, "langchain")
    assert all((dst / p).is_file() for p in DATABASE_PATHS)
    assert db_py.read_bytes() == original
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert_clean_tree(dst)


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_a_project_that_already_has_the_database_off_does_not_get_it_back_on_update(
    tmp_path, tagged_repo
):
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(tagged_repo, dst, "v100.0.0", database=False)
    git(dst, "init", "-q", "-b", "main")
    (dst / "backend/user_added.txt").write_text("mine\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")
    assert not any((dst / p).exists() for p in DATABASE_PATHS)

    update(dst, "v100.0.1")

    assert (answers(dst)["database"], answers(dst)["llm"]) == (False, "none")
    assert not any((dst / p).exists() for p in DATABASE_PATHS)
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert "A rule added by the second version." in (dst / ".claude/rules/fastapi.md").read_text()
    assert_clean_tree(dst)


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_both_parts_off_in_one_update_matches_a_fresh_minimal_render(
    tmp_path, tagged_repo
):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    lock = (dst / "backend/uv.lock").read_bytes()

    update(dst, "v100.0.1", database=False, storage=False)

    assert (answers(dst)["database"], answers(dst)["storage"]) == (False, False)
    assert_clean_tree(dst)
    after = files_of(dst)
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.1", database=False, storage=False)
    expected = files_of(fresh)
    after.pop("backend/user_added.txt")
    assert after.pop("backend/uv.lock") == lock
    expected.pop("backend/uv.lock")
    assert after == expected


GATEWAY_PATHS = [
    "backend/src/app/llm/langchain/gateway.py", "backend/src/app/llm/prices.py",
    "backend/src/app/llm/checkpoint_types.py", "backend/src/app/prompts/README.md",
    "backend/llm-contracts.toml", "backend/tests/models.py", "backend/tests/recording.py",
    "backend/tests/test_llm_gateway.py", "backend/tests/evals/conftest.py",
]
# What stays when only the gateway goes: the database and its migrations, models.py among them.
KEPT_WITHOUT_GATEWAY = [
    "backend/src/app/db.py", "backend/src/app/models.py", "backend/migrations/env.py",
    "backend/migrations/versions/20261026_0003_agent_checkpoints.py",
    "backend/compose/db/compose.yaml", "backend/tests/db.py", "backend/tests/test_db.py",
]


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_gateway_off_on_update_deletes_its_files_and_keeps_the_database(
    tmp_path, tagged_repo
):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    assert all((dst / p).is_file() for p in GATEWAY_PATHS)
    (dst / USER_MIGRATION).write_text("# my migration\n")
    (dst / "backend/src/app/product.py").write_text("# the product\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: product")
    lock = (dst / "backend/uv.lock").read_bytes()

    update(dst, "v100.0.1", llm="none")

    assert (answers(dst)["database"], answers(dst)["llm"]) == (True, "none")
    assert [p for p in GATEWAY_PATHS if (dst / p).exists()] == []
    assert all((dst / p).is_file() for p in KEPT_WITHOUT_GATEWAY)
    assert (dst / USER_MIGRATION).read_text() == "# my migration\n"
    assert (dst / "backend/src/app/product.py").read_text() == "# the product\n"
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert (dst / "backend/uv.lock").read_bytes() == lock
    assert_clean_tree(dst)
    after = files_of(dst)
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.1", llm="none")
    expected = files_of(fresh)
    for own in ("backend/user_added.txt", "backend/src/app/product.py", "backend/uv.lock",
                USER_MIGRATION):
        after.pop(own)
    expected.pop("backend/uv.lock")
    # The whole tree equals a fresh llm=none render, apart from the user's files.
    assert after == expected


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_gateway_off_deletes_its_files_even_if_the_user_edited_them(
    tmp_path, tagged_repo
):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    for rel in ("backend/src/app/llm/prices.py", "backend/src/app/llm/checkpoint_types.py",
                "backend/src/app/llm/redaction_rules.py", "backend/src/app/llm/core/budget.py"):
        path = dst / rel
        path.write_text(path.read_text() + "\n# my own change\n")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "feat: edit the gateway files")

    update(dst, "v100.0.1", llm="none")

    for rel in ("backend/src/app/llm/prices.py", "backend/src/app/llm/checkpoint_types.py",
                "backend/src/app/llm/redaction_rules.py", "backend/src/app/llm/core/budget.py"):
        assert not (dst / rel).exists(), rel
        assert "my own change" in git(dst, "show", f"HEAD:{rel}")  # recoverable


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_turning_the_gateway_back_on_restores_it(tmp_path, tagged_repo):
    dst = committed_project(tagged_repo, tmp_path / "project", "v100.0.0")
    lock = (dst / "backend/uv.lock").read_bytes()
    update(dst, "v100.0.1", llm="none")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: gateway off")
    assert not (dst / "backend/src/app/llm").exists()

    update(dst, "v100.0.1", llm="langchain")

    assert (answers(dst)["database"], answers(dst)["llm"]) == (True, "langchain")
    assert all((dst / p).is_file() for p in GATEWAY_PATHS)
    assert (dst / "backend/user_added.txt").read_text() == "mine\n"
    assert_clean_tree(dst)
    after = files_of(dst)
    fresh = tmp_path / "fresh"
    write_base(fresh)
    copy_layer(tagged_repo, fresh, "v100.0.1")
    after.pop("backend/user_added.txt")
    assert after.pop("backend/uv.lock") == lock
    expected = files_of(fresh)
    expected.pop("backend/uv.lock")
    assert after == expected


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
def test_the_gateway_cannot_be_turned_on_without_the_database(tmp_path, tagged_repo):
    dst = tmp_path / "project"
    write_base(dst)
    copy_layer(tagged_repo, dst, "v100.0.0", database=False)
    git(dst, "init", "-q", "-b", "main")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "chore: start")

    with pytest.raises(ValueError, match="langchain needs database: true"):
        update(dst, "v100.0.1", llm="langchain")


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
@pytest.mark.parametrize(("data", "gateway"), [({"database": True}, True),
                                                ({"database": True, "llm": "none"}, False)],
                         ids=["database-true-only", "with-llm-none"])
def test_a_v0_7_1_project_without_the_database_has_no_llm_answer_yet(tmp_path, data, gateway):
    # v0.7.1 recorded no `llm`, so the first update falls back to the default and
    # `-d database=true` alone brings the gateway back (the README and decision 0005 say so; pass
    # `-d llm=none` to keep it off). Once `llm` is recorded, database=true alone leaves it none.
    dst = committed_baseline_project(tmp_path / "project", "v0.7.1", database=False)
    assert "llm" not in answers(dst)

    copier.run_update(dst, answers_file=ANSWERS_FILE, data=data, defaults=True, unsafe=True,
                      overwrite=True, quiet=True, vcs_ref="HEAD")

    assert answers(dst)["llm"] == ("langchain" if gateway else "none")
    assert (dst / "backend/src/app/llm").exists() is gateway


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
@pytest.mark.parametrize(("database", "llm"), [(True, "langchain"), (False, "none")],
                         ids=["database-on", "database-off"])
def test_updating_a_v0_7_1_project_with_defaults_records_the_llm_answer_and_deletes_only_the_old_paths(
    tmp_path, database, llm
):
    # v0.7.1 has no `llm` answer. `copier update --defaults` gives such a project the default,
    # langchain with the database and none without it, and deletes only the old import paths: the `llm is defined`
    # guards in copier.yml's _exclude are for the old copy's answers, which have no `llm`. Goes
    # when the README stops supporting updates from 0.7.x.
    dst = committed_baseline_project(tmp_path / "project", "v0.7.1", database=database)
    assert "llm" not in yaml.safe_load((dst / ANSWERS_FILE).read_text())
    before = tracked(dst)
    lock = (dst / "backend" / "uv.lock").read_bytes()

    update_to_head(dst)

    after = files_of(dst)
    # Only what 0.8.0 removes on purpose, and only where the gateway exists.
    assert before - after.keys() == (REMOVED_FILES if database else set()), "unexpected deletions"
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["llm"] == llm
    assert not list(dst.rglob("*.rej"))
    assert [p for p, data in after.items() if b"<<<<<<<" in data] == []
    fresh = files_of(render_at(tmp_path / "fresh", "HEAD", database=database))
    after.pop("backend/user_added.txt")
    assert after.pop("backend/uv.lock") == lock
    fresh.pop("backend/uv.lock")
    assert after == fresh


@pytest.mark.filterwarnings("ignore::copier.errors.MissingFileWarning")
@pytest.mark.parametrize("database", [True, False], ids=["database-on", "database-off"])
def test_updating_a_v0_8_0_project_with_defaults_deletes_nothing_and_adds_migration_0006(
    tmp_path, database
):
    # README "Upgrading to 0.9.0": `copier update --defaults`, then `make migrate` for the new
    # revision 0006. v0.8.0 already records `llm`, so the answers file does not change, nothing is
    # deleted, the user's file stays, and the project equals a fresh render at HEAD (its own
    # lockfile aside: the user runs `make lock`). Goes when the README stops supporting 0.8.x.
    dst = committed_baseline_project(tmp_path / "project", "v0.8.0", database=database)
    llm = "langchain" if database else "none"
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["llm"] == llm
    before = tracked(dst)
    lock = (dst / "backend" / "uv.lock").read_bytes()
    migration = "backend/migrations/versions/20261110_0006_llm_call_redacted_count.py"

    update_to_head(dst)

    after = files_of(dst)
    assert before - after.keys() == set(), "update deleted files"
    assert (migration in after) is database, "revision 0006 arrives with the database"
    assert yaml.safe_load((dst / ANSWERS_FILE).read_text())["llm"] == llm
    assert (dst / "backend" / "user_added.txt").read_text() == "mine\n"
    assert not list(dst.rglob("*.rej"))
    assert [p for p, data in after.items() if b"<<<<<<<" in data] == []
    fresh = files_of(render_at(tmp_path / "fresh", "HEAD", database=database))
    after.pop("backend/user_added.txt")
    assert after.pop("backend/uv.lock") == lock
    fresh.pop("backend/uv.lock")
    assert after == fresh
