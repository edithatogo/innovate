"""Tests for refreshed external acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path

RECEIPTS = Path("docs/source/_static/registry_submission_receipts.json")
TARGET_INVENTORY = Path("docs/source/_static/external_submission_target_inventory.json")

PACKAGE_TARGETS = {
    "python_pypi": ("0.5.0", "published"),
    "typescript_npm": ("0.5.0", "published"),
    "rust_crates_io": ("0.5.0", "published"),
    "julia_general": ("0.5.0", "submitted_open_pr"),
    "go_modules": ("v0.5.0", "published"),
    "csharp_nuget": ("0.5.0", "published"),
    "r_r_universe": ("0.5.0", "indexed"),
}


def _receipts() -> dict[str, object]:
    return json.loads(RECEIPTS.read_text(encoding="utf-8"))


def _target_inventory() -> dict[str, object]:
    return json.loads(TARGET_INVENTORY.read_text(encoding="utf-8"))


def test_registry_receipts_have_fresh_package_manager_live_evidence() -> None:
    """Package-manager receipts should include current observed registry states."""
    receipts = _receipts()
    live = {entry["target_id"]: entry for entry in receipts["live_evidence"]}

    assert receipts["captured_at"].startswith("2026-06-16")
    assert set(live) >= set(PACKAGE_TARGETS)
    for target_id, (version, observed_state) in PACKAGE_TARGETS.items():
        entry = live[target_id]
        assert entry["observed_version"] == version
        assert entry["observed_state"] == observed_state
        assert entry["checked_at"].startswith("2026-06-16")
        assert entry["source_url"].startswith("https://")
        assert entry["source_kind"] in {"registry_api", "github_api", "go_proxy"}


def test_submitted_receipts_carry_live_evidence_references() -> None:
    """Submitted package-manager receipts should point to their live evidence entries."""
    submitted = {entry["target_id"]: entry for entry in _receipts()["submitted_targets"]}

    for target_id in PACKAGE_TARGETS:
        assert submitted[target_id]["live_evidence_id"] == target_id


def test_pending_external_targets_have_current_requirement_sources() -> None:
    """Deferred/review-ready targets should have exact current requirement sources."""
    inventory = _target_inventory()
    refresh = {entry["target_id"]: entry for entry in inventory["pending_requirement_refresh"]}
    non_submitted = {
        entry["target_id"]
        for entry in inventory["targets"]
        if entry["status"] in {"deferred", "ready_for_review", "ready_for_maintainer", "not_applicable"}
    }

    assert inventory["inventory_date"] == "2026-06-16"
    assert set(refresh) == non_submitted
    for target_id, entry in refresh.items():
        assert entry["checked_at"].startswith("2026-06-16"), target_id
        assert entry["owner"], target_id
        assert entry["action_boundary"], target_id
        assert entry["requirement_sources"], target_id
        assert all(str(url).startswith("https://") for url in entry["requirement_sources"]), target_id
        assert "generic blocker" not in entry["action_boundary"].lower()


def test_pending_external_targets_keep_specific_owner_backed_next_actions() -> None:
    """Target next-actions should avoid vague blocked language after refresh."""
    inventory = _target_inventory()
    targets = {
        entry["target_id"]: entry
        for entry in inventory["targets"]
        if entry["status"] in {"deferred", "ready_for_review", "ready_for_maintainer", "not_applicable"}
    }

    for target_id, entry in targets.items():
        text = " ".join([entry["owner"], entry["next_action"]]).lower()
        assert "blocked" not in text, target_id
        assert "maintainer" in text or "no submission claimed" in text, target_id
