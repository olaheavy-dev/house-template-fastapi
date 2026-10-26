"""This repository's own governance: the ruleset protecting main and the release setup."""

from __future__ import annotations

import json
import re

import yaml

from conftest import ROOT


def test_required_checks_are_jobs_the_workflows_run_on_pull_requests():
    ruleset = json.loads((ROOT / ".github/rulesets/main.json").read_text())
    rules = {r["type"]: r for r in ruleset["rules"]}
    required = {c["context"] for c in rules["required_status_checks"]["parameters"]["required_status_checks"]}
    names = set()
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        triggers = workflow[True] if True in workflow else workflow["on"]  # YAML 1.1 reads `on` as True
        if "pull_request" not in triggers:
            continue
        for job in workflow["jobs"].values():
            name = job["name"]
            if "${{ matrix.language }}" in name:
                names |= {name.replace("${{ matrix.language }}", lang) for lang in job["strategy"]["matrix"]["language"]}
            else:
                names.add(name)
    assert required <= names, required - names


def test_ruleset_gates_merges_on_code_scanning_and_squash_only():
    rules = {r["type"]: r for r in json.loads((ROOT / ".github/rulesets/main.json").read_text())["rules"]}
    assert {t["tool"] for t in rules["code_scanning"]["parameters"]["code_scanning_tools"]} == {"CodeQL", "zizmor"}
    assert rules["pull_request"]["parameters"]["allowed_merge_methods"] == ["squash"]


def test_releases_start_at_0_1_with_v_tags():
    # Releases are immutable, so the first one can't be renumbered after the fact.
    config = json.loads((ROOT / "release-please-config.json").read_text())
    assert config["include-v-in-tag"] and config["bump-minor-pre-major"]
    assert config["packages"]["."]["initial-version"] == "0.1.0"
    # release-please moves the manifest to each release, so only its shape is fixed.
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert manifest.keys() == {"."} and re.fullmatch(r"\d+\.\d+\.\d+", manifest["."])
