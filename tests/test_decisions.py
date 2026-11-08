"""The design records in docs/decisions/ and their register stay in step."""

from __future__ import annotations

import re

from conftest import ROOT

DECISIONS = ROOT / "docs" / "decisions"
RECORD = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")
ROW = re.compile(r"^\| \[(\d{4})\]\(([^)]+)\) \| .* \| (\S+(?: by \d{4})?) \|$")
STATUS = re.compile(r"^Status: (Accepted|Proposed|Superseded by \d{4})\b", re.M)
SECTIONS = ("## Context", "## Decision", "## Consequences", "## Alternatives considered", "## Sources")


def _rows() -> list[tuple[str, str, str]]:
    """(number, link target, status) for each row of the Records table in README.md, in order."""
    rows = []
    for line in (DECISIONS / "README.md").read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows.append((match.group(1), match.group(2), match.group(3)))
    return rows


def _files() -> list[str]:
    return sorted(p.name for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))


def test_every_numbered_file_is_a_well_named_record():
    assert _files(), "no records found"
    for name in _files():
        assert RECORD.match(name), name


def test_every_record_file_is_listed_in_the_register():
    listed = {target for _, target, _ in _rows()}
    assert set(_files()) <= listed, set(_files()) - listed


def test_every_register_row_has_a_file_and_a_matching_number():
    for number, target, _ in _rows():
        assert (DECISIONS / target).is_file(), target
        assert target.startswith(number + "-"), (target, number)


def test_register_rows_are_unique():
    targets = [target for _, target, _ in _rows()]
    assert len(targets) == len(set(targets)), "a record is listed twice"


def test_record_numbers_run_from_0001_without_gaps():
    numbers = [number for number, _, _ in _rows()]
    assert numbers == [f"{n:04d}" for n in range(1, len(numbers) + 1)], numbers


def test_each_heading_number_matches_the_filename():
    for number, target, _ in _rows():
        first = (DECISIONS / target).read_text().splitlines()[0]
        assert first.startswith(f"# {number}. "), (target, first)


def test_each_register_status_matches_the_record():
    for _, target, listed in _rows():
        match = STATUS.search((DECISIONS / target).read_text())
        assert match, target
        assert match.group(1).split()[0] == listed.split()[0], (target, listed, match.group(1))


def test_every_record_has_its_sections():
    for _, target, _ in _rows():
        text = (DECISIONS / target).read_text()
        for heading in SECTIONS:
            assert re.search(rf"^{heading}\b", text, re.M), (target, heading)
