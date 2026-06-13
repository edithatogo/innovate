"""Regression tests for Rust-native canonical operation completion."""

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
GAP_INVENTORY_PATH = Path("docs/source/_static/rust_native_operation_gap_inventory.json")
REQUIRED_PROMOTION_GATES = {
    "parity",
    "schema_compatibility",
    "error_mapping",
    "benchmark_evidence",
    "memory_evidence",
    "binding_smoke",
}


def _migration_entries() -> list[dict[str, Any]]:
    return json.loads(INVENTORY_PATH.read_text())["inventory"]


def _gap_inventory() -> dict[str, Any]:
    return json.loads(GAP_INVENTORY_PATH.read_text())


def test_operation_gap_inventory_covers_every_canonical_operation() -> None:
    """The track-local operation inventory should own every canonical operation."""
    gap_inventory = _gap_inventory()
    operations = {entry["operation"] for entry in gap_inventory["operation_gaps"]}

    assert set(gap_inventory["canonical_operations"]) == CANONICAL_OPERATIONS
    assert operations == CANONICAL_OPERATIONS
    assert gap_inventory["source_inventory"] == str(INVENTORY_PATH)


def test_every_canonical_operation_slice_has_terminal_ownership_state() -> None:
    """Canonical operations should have native/default or explicit boundary states."""
    allowed_terminal_states = {
        "native_default_guarded",
        "rust_native_promoted",
        "bridge_default_explicit_exception",
        "python_reference_boundary",
        "promoted_non_python_backend",
    }

    non_terminal = [
        (entry["operation"], entry["model_slice"], entry["promotion_state"])
        for entry in _migration_entries()
        if entry["operation"] in CANONICAL_OPERATIONS
        and entry["promotion_state"] not in allowed_terminal_states
    ]

    assert non_terminal == []


def test_native_default_slices_have_complete_promotion_evidence() -> None:
    """Native-default slices cannot keep required evidence gates unresolved."""
    unresolved: list[tuple[str, str, str, str]] = []
    for entry in _migration_entries():
        if entry["operation"] not in CANONICAL_OPERATIONS:
            continue
        if entry["current_owner"] != "rust_native":
            continue
        if entry["fallback_status"] != "native_default_no_fallback_needed":
            continue

        gates = entry["promotion_gates"]
        assert set(gates) >= REQUIRED_PROMOTION_GATES
        for gate_name in REQUIRED_PROMOTION_GATES:
            gate = gates[gate_name]
            if gate["required_before_default"] and gate["status"] != "passed":
                unresolved.append(
                    (
                        entry["operation"],
                        entry["model_slice"],
                        gate_name,
                        gate["status"],
                    )
                )

    assert unresolved == []

