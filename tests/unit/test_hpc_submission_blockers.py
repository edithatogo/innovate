"""Tests for the HPC submission blocker bundle."""

from __future__ import annotations

import json
from pathlib import Path

BLOCKERS_PATH = Path("docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json")


def test_hpc_submission_blockers_exist() -> None:
    assert BLOCKERS_PATH.is_file()


def test_hpc_submission_blockers_cover_all_targets() -> None:
    blockers = json.loads(BLOCKERS_PATH.read_text(encoding="utf-8"))
    targets = {entry["target_id"]: entry for entry in blockers["blockers"]}
    resolved = {entry["target_id"]: entry for entry in blockers["resolved_blockers"]}

    assert blockers["schema_version"] == 1
    assert targets == {}
    assert set(resolved) == {"hpsf", "e4s"}
    for entry in resolved.values():
        assert entry["status"] == "ready_for_maintainer"
        assert entry["resolution"]
