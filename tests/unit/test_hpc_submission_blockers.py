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

    assert blockers["schema_version"] == 1
    assert set(targets) == {"spack", "easybuild", "hpsf", "e4s"}
    for entry in targets.values():
        assert entry["status"] == "blocked"
        assert entry["reason"]
