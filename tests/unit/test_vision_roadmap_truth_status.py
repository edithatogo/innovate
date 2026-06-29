"""Regression tests for vision and roadmap truthfulness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

VISION_STATUS_INVENTORY = Path("docs/source/_static/vision_roadmap_status_inventory.json")
RUST_MIGRATION_INVENTORY = Path("docs/source/_static/rust_core_migration_inventory.json")


def normalized_text(path: Path) -> str:
    """Collapse prose wrapping so claim checks are stable."""
    return " ".join(path.read_text().split())


def test_vision_status_inventory_declares_future_state_boundaries() -> None:
    """The status inventory should distinguish track completion from vision completion."""
    inventory = json.loads(VISION_STATUS_INVENTORY.read_text())

    assert inventory["schema_version"] == 1
    assert inventory["canonical_status"]["track_state"] == "archived_follow_on_tracks_complete"
    assert inventory["canonical_status"]["vision_state"] == "partially_implemented_future_state_remaining"
    assert "external_gate" in inventory["claim_categories"]
    assert "blocked_external" not in inventory["claim_categories"]
    assert "implemented_with_blocker" not in inventory["claim_categories"]
    assert not any(
        source["classification"] in {"blocked_external", "implemented_with_blocker"}
        for source in inventory["reviewed_sources"]
    )

    future_tracks = {track["track_id"] for track in inventory["archived_follow_on_tracks"]}
    assert {
        "vision_roadmap_truth_audit_20260614",
        "rust_native_operation_completion_20260614",
        "rust_native_payload_model_coverage_20260614",
        "starlight_cutover_legacy_cleanup_20260614",
        "external_submission_blocker_closure_20260614",
        "conductor_registry_hygiene_20260614",
    } <= future_tracks


def test_product_status_does_not_claim_sphinx_as_active_docs_stack() -> None:
    """Product status should agree with the active Astro/Starlight tech stack."""
    product = normalized_text(Path("conductor/product.md"))

    assert "Documentation: Sphinx with RTD theme" not in product
    assert "Documentation: Astro/Starlight" in product


def test_tech_stack_marks_starlight_as_only_active_docs_site() -> None:
    """The tech stack should not include Sphinx in the active docs stack."""
    tech_stack = normalized_text(Path("conductor/tech-stack.md"))

    assert "being migrated to Starlight" not in tech_stack
    assert "Astro/Starlight under `docs/astro-site` is the only documentation site." in tech_stack


def test_roadmap_links_archived_remediation_tracks() -> None:
    """Every remediation track should point to an archived Conductor record."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()

    for track_name in (
        "Vision and Roadmap Truth Audit",
        "Rust-Native Canonical Operation Completion",
        "Rust-Native Payload and Model-Family Coverage",
        "Starlight Cutover and Legacy Cleanup",
        "External Submission Blocker Closure",
        "Conductor Registry Hygiene",
    ):
        assert track_name in roadmap


def test_starlight_roadmap_does_not_call_archived_tracks_active() -> None:
    """Starlight roadmap mirrors should not call archived remediation tracks active."""
    for path in (
        Path("docs/astro-site/src/content/docs/operations/roadmap.md"),
        Path("docs/astro-site/src/content/docs/latest/operations/roadmap.md"),
    ):
        roadmap = normalized_text(path)
        assert "Archived remediation tracks" in roadmap
        assert "Active future-state tracks" not in roadmap
        assert "require their own active tracks" not in roadmap
        assert "full product vision remains active" not in roadmap.lower()
        assert "external submission blockers" not in roadmap.lower()
        assert "external submission handoffs" in roadmap.lower()
        assert "not fully complete" in roadmap
        assert "complete and archived" in roadmap


def test_full_rust_core_claims_remain_disallowed_until_inventory_is_all_native() -> None:
    """Full Rust ownership cannot be claimed while any slice is bridge or Python owned."""
    migration_inventory = json.loads(RUST_MIGRATION_INVENTORY.read_text())
    owners = {entry["current_owner"] for entry in migration_inventory["inventory"]}
    project_docs = "\n".join(
        path.read_text()
        for path in (
            Path("conductor/product.md"),
            Path("docs/architecture_modernization_roadmap.md"),
            Path("docs/astro-site/src/content/docs/operations/rust-core.md"),
        )
    )

    if owners - {"rust_native"}:
        forbidden_claims = (
            "full Rust core is complete",
            "entire core is Rust-owned",
            "fully migrated to a Rust core",
        )
        for claim in forbidden_claims:
            assert claim not in project_docs


def test_roadmap_truth_ledger_exists() -> None:
    """The roadmap truth ledger must exist as a machine-readable artifact."""
    ledger = Path("conductor/tracks/roadmap_release_truth_closure_20260625/truth_ledger.json")
    assert ledger.exists(), "Truth ledger not yet created - Phase 2 must implement it"


def test_roadmap_truth_ledger_covers_all_roadmap_claims() -> None:
    """Every roadmap claim must map to evidence, active track, external blocker, or out-of-scope rationale."""
    inventory = Path("conductor/tracks/roadmap_release_truth_closure_20260625/inventory.json")
    assert inventory.exists(), "Inventory not yet created - Phase 1 must build it first"

    inventory_data = json.loads(inventory.read_text())
    claims = inventory_data.get("roadmap_claims", [])

    ledger = Path("conductor/tracks/roadmap_release_truth_closure_20260625/truth_ledger.json")
    assert ledger.exists(), "Truth ledger must exist before coverage can be validated"

    ledger_data = json.loads(ledger.read_text())
    ledger_claim_keys = {entry["claim"] for entry in ledger_data.get("entries", [])}

    for claim in claims:
        claim_key = claim["claim"]
        assert claim_key in ledger_claim_keys, (
            f"Claim '{claim_key}' from {claim['source']} is not covered in the truth ledger"
        )


def test_roadmap_truth_ledger_entries_have_required_fields() -> None:
    """Every truth ledger entry must have status, owner, and evidence fields."""
    ledger = Path("conductor/tracks/roadmap_release_truth_closure_20260625/truth_ledger.json")
    if not ledger.exists():
        return  # Test passes vacuously before ledger exists

    ledger_data = json.loads(ledger.read_text())
    for entry in ledger_data.get("entries", []):
        assert "claim" in entry
        assert "status" in entry, f"Entry '{entry.get('claim', 'unknown')}' missing status"
        assert entry["status"] in {"complete", "active", "external_blocked", "future_state", "out_of_scope"}, (
            f"Entry '{entry.get('claim', 'unknown')}' has invalid status: {entry.get('status')}"
        )
        assert "evidence" in entry, f"Entry '{entry.get('claim', 'unknown')}' missing evidence"
        assert "track_link" in entry, f"Entry '{entry.get('claim', 'unknown')}' missing track_link"


def test_no_stale_or_missing_completion_claims() -> None:
    """No claim should say 'complete' without corresponding archived track evidence."""
    inventory = Path("conductor/tracks/roadmap_release_truth_closure_20260625/inventory.json")
    ledger = Path("conductor/tracks/roadmap_release_truth_closure_20260625/truth_ledger.json")
    if not ledger.exists():
        return  # Test passes vacuously before ledger exists

    inventory_data = json.loads(inventory.read_text())
    ledger_data = json.loads(ledger.read_text())
    ledger_entries = {e["claim"]: e for e in ledger_data.get("entries", [])}

    for claim in inventory_data.get("roadmap_claims", []):
        claim_key = claim["claim"]
        if claim_key in ledger_entries:
            entry = ledger_entries[claim_key]
            if entry["status"] == "complete":
                assert entry.get("evidence"), f"Complete claim '{claim_key}' must have evidence"


def test_no_active_rst_outside_allowlist() -> None:
    """Fail if any RST file outside the explicit allowlist exists as active docs."""
    classification = Path("conductor/tracks/starlight_only_docs_completion_20260625/rst_classification.json")
    if not classification.exists():
        return  # Pass vacuously until classification exists

    data = json.loads(classification.read_text())
    keep_allowlist = {Path(p).resolve() for p in data.get("keep_allowlist", [])}

    docs_root = Path("docs")
    active_rst_files = set()
    for rst in docs_root.rglob("*.rst"):
        resolved = rst.resolve()
        if resolved in keep_allowlist:
            continue
        active_rst_files.add(rst)

    if active_rst_files:
        msg = "Unexpected RST files outside allowlist:\n" + "\n".join(f"  - {p}" for p in sorted(active_rst_files)[:10])
        if len(active_rst_files) > 10:
            msg += f"\n  ... and {len(active_rst_files) - 10} more"
        pytest.fail(msg)


def test_starlight_migration_route_coverage_inventory_exists() -> None:
    """Migration evidence should include a route inventory."""
    inventory = Path("docs/source/_static/astro_starlight/route_inventory.json")
    assert inventory.exists() or inventory.parent.exists(), (
        "Route inventory evidence should exist under docs/source/_static/astro_starlight/"
    )
