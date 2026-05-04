"""Tests for the machine-readable Rust core migration inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_OPERATIONS = {
    "discover_models",
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
}
INVENTORY_PATH = Path("docs/source/_static/rust_core_migration_inventory.json")
DOSSIER_PATH = Path("docs/source/_static/rust_core_promotion_dossier_example.json")
ROADMAP_PATH = Path("docs/source/rust_core_roadmap.rst")
REQUIRED_ENTRY_FIELDS = {
    "operation",
    "model_slice",
    "current_owner",
    "fallback_status",
    "native_scope",
    "fallback_scope",
    "python_reference_scope",
    "profiling_requirements",
    "promotion_blockers",
}


def load_inventory() -> dict[str, Any]:
    """Load the Rust migration inventory artifact."""
    return json.loads(INVENTORY_PATH.read_text())


def load_dossier() -> dict[str, Any]:
    """Load the Rust promotion dossier example artifact."""
    return json.loads(DOSSIER_PATH.read_text())


def normalized_text(path: Path) -> str:
    """Read prose with line wrapping collapsed for stable phrase assertions."""
    return " ".join(path.read_text().split())


def test_rust_migration_inventory_declares_required_fields_and_enums() -> None:
    """Each inventory row should use the documented field names and enum values."""
    inventory = load_inventory()

    assert inventory["schema_version"] == 1
    assert set(inventory["owner_values"]) == {"rust_native", "python_bridge", "python_reference"}
    assert set(inventory["fallback_status_values"]) == {
        "native_default_no_fallback_needed",
        "native_default_python_bridge_fallback",
        "python_bridge_default",
        "python_reference_only",
    }
    assert inventory["inventory"]

    owner_values = set(inventory["owner_values"])
    fallback_values = set(inventory["fallback_status_values"])
    for entry in inventory["inventory"]:
        assert set(entry) >= REQUIRED_ENTRY_FIELDS
        assert entry["current_owner"] in owner_values
        assert entry["fallback_status"] in fallback_values
        assert isinstance(entry["profiling_requirements"], list)
        assert isinstance(entry["promotion_blockers"], list)
        assert all(isinstance(requirement, str) and requirement for requirement in entry["profiling_requirements"])
        assert all(isinstance(blocker, str) and blocker for blocker in entry["promotion_blockers"])


def test_rust_migration_inventory_covers_canonical_operations_and_mixed_ownership() -> None:
    """The inventory should cover all canonical operations without claiming full Rust ownership."""
    inventory = load_inventory()
    entries = inventory["inventory"]
    operations = {entry["operation"] for entry in entries}
    owners = {entry["current_owner"] for entry in entries}
    fallback_statuses = {entry["fallback_status"] for entry in entries}

    assert operations >= CANONICAL_OPERATIONS
    assert "rust_native" in owners
    assert "python_bridge" in owners
    assert "python_reference" in owners
    assert "native_default_python_bridge_fallback" in fallback_statuses
    assert "python_bridge_default" in fallback_statuses
    assert any(entry["current_owner"] == "rust_native" for entry in entries)
    assert any(entry["current_owner"] == "python_bridge" for entry in entries)
    assert any(entry["operation"] == "all_kernel_operations" for entry in entries)


def test_rust_migration_inventory_records_native_and_fallback_slices() -> None:
    """The inventory should include concrete Rust-native slices and Python bridge slices."""
    entries = load_inventory()["inventory"]
    native_slices = {
        (entry["operation"], entry["model_slice"]) for entry in entries if entry["current_owner"] == "rust_native"
    }
    bridge_slices = {
        (entry["operation"], entry["model_slice"]) for entry in entries if entry["current_owner"] == "python_bridge"
    }

    assert ("discover_models", "all_packaged_discovery_metadata") in native_slices
    assert ("predict_model", "bass_simple_fitted_state") in native_slices
    assert ("simulate_model", "bass_simple_fitted_state") in native_slices
    assert ("fit_model", "bass_and_other_model_families") in bridge_slices
    assert ("predict_model", "other_model_families_or_unsupported_payloads") in bridge_slices
    assert ("diagnose_model", "bass_and_other_model_families") in bridge_slices

    for operation, _model_slice in bridge_slices:
        assert operation in CANONICAL_OPERATIONS


def test_rust_migration_inventory_profiles_required_promotion_evidence_categories() -> None:
    """Inventory and roadmap should preserve the promotion dossier evidence categories."""
    entries = load_inventory()["inventory"]
    requirements = " ".join(requirement for entry in entries for requirement in entry["profiling_requirements"])
    blockers = " ".join(blocker for entry in entries for blocker in entry["promotion_blockers"])
    roadmap = normalized_text(ROADMAP_PATH)

    for phrase in (
        "Criterion benchmark coverage",
        "CPU flamegraph",
        "Memory profiling",
        "Rust CPU timing",
        "bridge fallback rate",
        "XLA eligibility",
    ):
        assert phrase in requirements + " " + blockers

    for phrase in (
        "promotion dossier",
        "Criterion output for Rust-native CPU paths",
        "Python reference timings",
        "XLA compile cost and steady-state runtime when eligible",
        "memory evidence for allocation-sensitive slices",
        "regression threshold that CI or release checks can enforce",
    ):
        assert phrase in roadmap


def test_rust_promotion_dossier_example_records_required_evidence_sections() -> None:
    """The promotion dossier example should preserve the required evidence structure."""
    dossier = load_dossier()

    assert dossier["schema_version"] == 1
    assert dossier["dossier_type"] == "rust_core_promotion_dossier"
    assert dossier["evidence_state"] == "template_example_not_release_evidence"
    assert dossier["slice"]["operation"] in CANONICAL_OPERATIONS
    assert dossier["slice"]["current_owner"] in {"rust_native", "python_bridge", "python_reference"}
    assert dossier["slice"]["target_owner"] in {"rust_native", "python_bridge", "python_reference"}
    assert dossier["slice"]["fallback_status"] in {
        "native_default_no_fallback_needed",
        "native_default_python_bridge_fallback",
        "python_bridge_default",
        "python_reference_only",
    }

    required_sections = {
        "promotion_decision",
        "parity",
        "schema_compatibility",
        "error_mapping",
        "cpu_benchmark",
        "memory_profiling",
        "xla_gpu_eligibility",
        "fallback_rate",
        "binding_smoke",
        "release_record",
    }
    assert required_sections <= set(dossier)

    assert "regression_threshold" in dossier["cpu_benchmark"]["metrics"]
    assert "allocation_regression_threshold" in dossier["memory_profiling"]["metrics"]
    assert "XLA compile cost" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "XLA steady-state runtime" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "Rust-native CPU runtime" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "python_bridge_fallback_requests" in dossier["fallback_rate"]["metrics"]
    assert {"rust", "python"} <= set(dossier["binding_smoke"]["required_bindings"])
