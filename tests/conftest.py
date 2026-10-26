"""Shared fixtures: render this layer into a temporary project that already has base answers.

The generated-project CI job applies the real house-template-base first; these tests only need
the answers file the base layer leaves, so they write one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import copier
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "template"

BASE_ANSWERS = {
    "_src_path": "gh:olaheavy-dev/house-template-base",
    "_commit": "v0.2.1",
    "project_name": "Course Lens",
    "project_slug": "course-lens",
    "description": "Turns a course into something you can search, question and practise.",
    "github_org": "olaheavy-dev",
    "domain": "courselens.example.dev",
    "stacks": ["fastapi", "nextjs"],
    "container_images": True,
}


@pytest.fixture
def project(tmp_path: Path) -> Callable[..., Path]:
    """A project directory holding base answers (pass base=None for none, or overrides)."""

    def _project(base: dict[str, object] | None = BASE_ANSWERS, **overrides: object) -> Path:
        dst = tmp_path / "project"
        dst.mkdir(exist_ok=True)
        if base is not None:
            answers = {**base, **overrides}
            (dst / ".copier-answers.base.yml").write_text(yaml.safe_dump(answers))
        return dst

    return _project


@pytest.fixture
def render(project: Callable[..., Path]) -> Callable[..., Path]:
    """Render the layer (including uncommitted changes) into a project with base answers."""

    def _render(dst: Path | None = None, **data: object) -> Path:
        dst = dst or project()
        copier.run_copy(str(ROOT), dst, data=data, defaults=True, unsafe=True, quiet=True,
                        vcs_ref="HEAD")
        return dst

    return _render
