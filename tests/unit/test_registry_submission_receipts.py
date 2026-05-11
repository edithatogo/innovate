"""Tests for the registry submission receipt bundle."""

from __future__ import annotations

import json
from pathlib import Path

RECEIPTS_PATH = Path("docs/source/_static/registry_submission_receipts.json")

SUBMITTED_TARGETS = {
    "python_pypi": "https://pypi.org/project/innovate/",
    "typescript_npm": "https://www.npmjs.com/package/innovate.ts",
    "rust_crates_io": "https://crates.io/crates/innovate-rs",
    "julia_general": "https://github.com/JuliaRegistries/General/pull/155126",
    "go_modules": "https://pkg.go.dev/github.com/edithatogo/innovate/bindings/go",
    "csharp_nuget": "https://www.nuget.org/packages/innovate.cs/",
    "r_r_universe": "https://github.com/edithatogo/edithatogo.r-universe.dev/commit/8033021d768d9c8917f711f30b3387b2ef65b90b",
}


def load_receipts() -> dict[str, object]:
    return json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))


def test_registry_submission_receipts_file_exists() -> None:
    """The receipt bundle should be present as a durable evidence artifact."""
    assert RECEIPTS_PATH.is_file()


def test_registry_submission_receipts_cover_live_targets() -> None:
    """All live package-manager targets should have concrete receipt URLs."""
    receipts = load_receipts()

    assert receipts["schema_version"] == 1
    submitted = {
        entry["target_id"]: entry
        for entry in receipts["submitted_targets"]
    }
    assert set(submitted) == set(SUBMITTED_TARGETS)

    for target_id, receipt_url in SUBMITTED_TARGETS.items():
        entry = submitted[target_id]
        assert entry["receipt_url"] == receipt_url
        assert entry["version"]
        assert entry["evidence"]


def test_registry_submission_receipts_record_pending_targets() -> None:
    """R/CRAN and HPC targets should remain explicitly labeled when blocked."""
    receipts = load_receipts()

    pending = {entry["target_id"]: entry for entry in receipts["pending_targets"]}

    assert pending["r_cran"]["status"] == "deferred"
    assert pending["spack"]["status"] == "blocked"
    assert pending["easybuild"]["status"] == "blocked"
    assert pending["hpsf"]["status"] == "blocked"
    assert pending["e4s"]["status"] == "blocked"
