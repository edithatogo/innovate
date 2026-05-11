"""Tests for the registry submission execution track."""

from __future__ import annotations

import json
from pathlib import Path

INDEX_PATH = Path("docs/source/index.rst")
TRACKS_PATH = Path("conductor/tracks.md")
INVENTORY_PATH = Path("docs/source/_static/registry_submission_inventory.json")
TRACK_PLAN = Path("conductor/tracks/registry_submission_execution_20260511/plan.md")

PACKAGE_TARGETS = {
    "python_pypi",
    "typescript_npm",
    "rust_crates_io",
    "r_r_universe",
    "r_cran",
    "julia_general",
    "go_modules",
    "csharp_nuget",
}

HPC_TARGETS = {
    "spack",
    "easybuild",
    "hpsf",
    "e4s",
}


def load_inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_registry_submission_track_is_registered() -> None:
    """The new submission track should be discoverable from the registry."""
    assert TRACK_PLAN.is_file()
    registry = TRACKS_PATH.read_text(encoding="utf-8")

    assert "registry_submission_execution_20260511" in registry
    assert "Package and HPC Registry Submission Execution" in registry


def test_registry_submission_inventory_covers_package_and_hpc_targets() -> None:
    """The submission inventory should enumerate every target registry."""
    inventory = load_inventory()

    assert inventory["schema_version"] == 1
    assert set(inventory["package_targets"]) == PACKAGE_TARGETS
    assert set(inventory["hpc_targets"]) == HPC_TARGETS


def test_registry_submission_inventory_requires_audit_fields() -> None:
    """Every inventory row should record status, owner, and evidence."""
    inventory = load_inventory()

    for entry in inventory["targets"]:
        assert entry["target_id"]
        assert entry["surface"]
        assert entry["registry"]
        assert entry["owner"]
        assert entry["submission_status"] in {
            "submitted",
            "deferred",
            "blocked",
            "not_applicable",
        }
        assert entry["evidence"], entry["target_id"]
        assert entry["receipt"] is not None
        assert entry["release_path"], entry["target_id"]


def test_registry_submission_docs_still_reflect_readiness_not_submission() -> None:
    """Existing docs should not overstate submission until receipts exist."""
    binding_docs = Path("docs/source/binding_publication_ci.rst").read_text(encoding="utf-8")
    hpc_docs = Path("docs/source/hpc_packaging_registry_readiness.rst").read_text(encoding="utf-8")
    community_docs = Path("docs/source/community_submission_readiness.rst").read_text(encoding="utf-8")

    assert "publication targets" in binding_docs.lower()
    assert "registry_submission_receipts" in binding_docs
    assert "readiness planning" in hpc_docs.lower()
    assert "not a registry claim" in hpc_docs.lower()
    assert "No submission claims readiness without evidence" in community_docs
