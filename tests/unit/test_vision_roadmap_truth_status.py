"""Regression tests for vision and roadmap truthfulness."""

from __future__ import annotations

import json
from pathlib import Path

VISION_STATUS_INVENTORY = Path("docs/source/_static/vision_roadmap_status_inventory.json")
RUST_MIGRATION_INVENTORY = Path("docs/source/_static/rust_core_migration_inventory.json")


def normalized_text(path: Path) -> str:
    """Collapse prose wrapping so claim checks are stable."""
    return " ".join(path.read_text().split())


def test_vision_status_inventory_declares_future_state_boundaries() -> None:
    """The status inventory should distinguish track completion from vision completion."""
    inventory = json.loads(VISION_STATUS_INVENTORY.read_text())

    assert inventory["schema_version"] == 1
    assert inventory["canonical_status"]["track_state"] == "active_follow_on_tracks_registered"
    assert inventory["canonical_status"]["vision_state"] == "partially_implemented_future_state_remaining"

    future_tracks = {track["track_id"] for track in inventory["future_state_tracks"]}
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


def test_tech_stack_marks_sphinx_as_legacy_not_active_migration() -> None:
    """The tech stack should not imply the Starlight migration is still active."""
    tech_stack = normalized_text(Path("conductor/tech-stack.md"))

    assert "being migrated to Starlight" not in tech_stack
    assert "Legacy Sphinx source" in tech_stack


def test_roadmap_links_remaining_future_state_tracks() -> None:
    """Every unresolved vision boundary should point to a granular Conductor track."""
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


def test_full_rust_core_claims_remain_blocked_until_inventory_is_all_native() -> None:
    """Full Rust ownership cannot be claimed while any slice is bridge or Python owned."""
    migration_inventory = json.loads(RUST_MIGRATION_INVENTORY.read_text())
    owners = {entry["current_owner"] for entry in migration_inventory["inventory"]}
    project_docs = "\n".join(
        path.read_text()
        for path in (
            Path("conductor/product.md"),
            Path("docs/architecture_modernization_roadmap.md"),
            Path("docs/source/rust_core_roadmap.rst"),
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
