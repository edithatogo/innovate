"""Tests for external acceptance receipts and owner-backed deferrals."""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_PATH = Path("docs/source/_static/external_acceptance_deferrals.json")
TARGET_INVENTORY = Path("docs/source/_static/external_submission_target_inventory.json")
RECEIPTS = Path("docs/source/_static/registry_submission_receipts.json")


def load_ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def load_inventory() -> dict[str, object]:
    return json.loads(TARGET_INVENTORY.read_text(encoding="utf-8"))


def load_receipts() -> dict[str, object]:
    return json.loads(RECEIPTS.read_text(encoding="utf-8"))


def test_external_acceptance_ledger_covers_all_targets() -> None:
    """Every external target should be classified as receipt or owner-backed deferral."""
    ledger = load_ledger()
    inventory = load_inventory()
    ledger_targets = {entry["target_id"] for entry in ledger["receipts"] + ledger["owner_backed_deferrals"]}
    inventory_targets = {entry["target_id"] for entry in inventory["targets"]}

    assert ledger["schema_version"] == 1
    assert ledger["captured_at"].startswith("2026-06-16")
    assert ledger["claim_policy"].startswith("Do not claim acceptance")
    assert ledger_targets == inventory_targets


def test_receipts_match_submitted_registry_receipts() -> None:
    """Submitted package-manager targets should carry receipt evidence, not deferrals."""
    ledger = load_ledger()
    receipts = {entry["target_id"]: entry for entry in load_receipts()["submitted_targets"]}
    ledger_receipts = {entry["target_id"]: entry for entry in ledger["receipts"]}

    assert set(ledger_receipts) == set(receipts)
    for target_id, entry in ledger_receipts.items():
        assert entry["receipt_url"] == receipts[target_id]["receipt_url"]
        assert entry["receipt_kind"] in {"registry_publication", "open_registry_pr", "module_index"}
        assert entry["acceptance_state"] in {"published", "submitted_open_pr", "indexed"}


def test_deferrals_are_owner_backed_and_receipt_gated() -> None:
    """Ready/deferred/not-applicable targets should have exact owner-backed states."""
    ledger = load_ledger()
    inventory = {entry["target_id"]: entry for entry in load_inventory()["targets"]}

    for entry in ledger["owner_backed_deferrals"]:
        target = inventory[entry["target_id"]]
        assert target["status"] in {"deferred", "ready_for_review", "ready_for_maintainer", "not_applicable"}
        assert entry["owner"] == target["owner"]
        assert entry["current_state"] == target["status"]
        assert entry["prepared_packet"], entry["target_id"]
        assert entry["external_action_url"].startswith("https://"), entry["target_id"]
        assert entry["receipt_or_revisit_condition"], entry["target_id"]
        assert "blocked" not in json.dumps(entry).lower(), entry["target_id"]


def test_inventory_references_external_acceptance_ledger() -> None:
    """The target inventory should point to the receipt/deferral ledger."""
    inventory = load_inventory()

    assert "docs/source/_static/external_acceptance_deferrals.json" in inventory["source_artifacts"]
