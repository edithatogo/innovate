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
