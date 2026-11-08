"""The design records in docs/decisions/ and their register stay in step."""

from __future__ import annotations

import re

from conftest import ROOT

DECISIONS = ROOT / "docs" / "decisions"
RECORD = re.compile(r"^\d{4}-[a-z0-9-]+\.md$")
ROW = re.compile(r"^\| \[(\d{4})\]\(([^)]+)\) \|")


def _register_rows() -> dict[str, str]:
    """Map each record row's link target to its number, from the Records table in README.md."""
    rows: dict[str, str] = {}
    for line in (DECISIONS / "README.md").read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows[match.group(2)] = match.group(1)
    return rows


def test_every_record_file_is_listed_in_the_register():
    files = {p.name for p in DECISIONS.glob("*.md") if RECORD.match(p.name)}
    assert files, "no records found"
    assert files <= set(_register_rows()), files - set(_register_rows())


def test_every_register_row_has_a_file_and_a_matching_number():
    for target, number in _register_rows().items():
        assert (DECISIONS / target).is_file(), target
        assert target.startswith(number + "-"), (target, number)


def test_record_numbers_are_unique_and_in_order():
    numbers = [number for number in _register_rows().values()]
    assert numbers == sorted(set(numbers))


def test_every_record_states_a_status_and_its_sections():
    for target in _register_rows():
        text = (DECISIONS / target).read_text()
        assert re.search(r"^Status: (Accepted|Proposed|Superseded by \d{4})", text, re.M), target
        for heading in ("## Context", "## Decision", "## Consequences", "## Alternatives considered", "## Sources"):
            assert re.search(rf"^{heading}", text, re.M), (target, heading)
