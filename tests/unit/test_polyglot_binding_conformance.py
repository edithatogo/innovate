"""Tests for the polyglot binding conformance inventory."""

from __future__ import annotations

import json
from pathlib import Path

INVENTORY_PATH = Path("docs/source/_static/binding_conformance_inventory.json")
REQUIRED_LANGUAGES = {"python", "r", "julia", "typescript", "go", "csharp", "rust"}
REQUIRED_OPERATIONS = {
    "discover_models",
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
}
PROMOTED_RUST_MODELS = {"bass", "fisher_pry", "gompertz", "logistic", "norton_bass"}
PROMOTED_RUST_OPERATIONS = {
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
}
RUST_BINDING_SMOKE_PATH = Path("docs/source/_static/rust_native_operation_binding_smoke.json")


def _inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_binding_conformance_inventory_covers_all_languages() -> None:
    """Every roadmap binding should have machine-readable conformance status."""
    payload = _inventory()

    assert payload["schema_version"] == "binding_conformance.v1"
    assert payload["kernel_schema_version"] == "1.0"
    bindings = {entry["language"]: entry for entry in payload["bindings"]}
    assert set(bindings) == REQUIRED_LANGUAGES
    assert all(entry["version"] == "0.5.0" for entry in bindings.values())


def test_binding_conformance_inventory_has_required_contract_fields() -> None:
    """Binding entries should expose status, capabilities, payloads, errors, and evidence."""
    for entry in _inventory()["bindings"]:
        assert entry["status"] in {"supported", "experimental", "limited"}
        assert entry["capabilities"]
        assert set(entry["operations"]).issuperset(REQUIRED_OPERATIONS)
        assert set(entry["payloads"]) >= {"KernelRequest", "KernelResponse", "KernelError"}
        assert set(entry["errors"]) >= {
            "invalid_request",
            "invalid_schema_version",
            "unavailable_model",
            "unsupported_operation",
            "backend_unavailable",
            "invalid_payload",
            "internal_error",
        }
        assert entry["package_checks"]
        assert entry["evidence_paths"]


def test_binding_conformance_inventory_points_to_existing_files() -> None:
    """Evidence and package metadata references should resolve in the repository."""
    for entry in _inventory()["bindings"]:
        for path in [*entry["evidence_paths"], entry["package_manifest"]]:
            assert Path(path).exists(), path


def test_binding_conformance_inventory_declares_promoted_rust_dispatch() -> None:
    """Bindings should declare whether promoted Rust operations are native or bridge-visible."""
    for entry in _inventory()["bindings"]:
        dispatch = entry["rust_dispatch"]

        assert set(dispatch["promoted_models"]) == PROMOTED_RUST_MODELS
        assert set(dispatch["promoted_operations"]) == PROMOTED_RUST_OPERATIONS
        assert dispatch["dispatch_mode"] in {"rust_native", "python_bridge_visible", "explicit_non_support"}
        assert dispatch["unsupported_payload_policy"] in {
            "structured_unsupported_native_operation",
            "bridge_error_contract",
        }
        assert dispatch["evidence_paths"]
        assert all(Path(path).exists() for path in dispatch["evidence_paths"])

        if entry["language"] == "rust":
            assert dispatch["dispatch_mode"] == "rust_native"
            assert dispatch["fallback_policy"] == "python_bridge_for_explicit_non_native_boundaries"
        else:
            assert dispatch["dispatch_mode"] == "python_bridge_visible"
            assert dispatch["fallback_policy"] == "language_client_uses_shared_kernel_bridge"


def test_rust_binding_smoke_matrix_covers_promoted_dispatch_metadata() -> None:
    """Rust operation smoke evidence should name promoted operation/model dispatch for each binding."""
    payload = json.loads(RUST_BINDING_SMOKE_PATH.read_text(encoding="utf-8"))

    assert set(payload["promoted_models"]) == PROMOTED_RUST_MODELS
    assert set(payload["promoted_operations"]) == PROMOTED_RUST_OPERATIONS
    assert set(payload["required_bindings"]) == REQUIRED_LANGUAGES

    by_binding = {entry["binding"]: entry for entry in payload["results"]}
    assert set(by_binding) == REQUIRED_LANGUAGES
    for binding, entry in by_binding.items():
        assert entry["status"] == "passed"
        assert set(entry["promoted_models"]) == PROMOTED_RUST_MODELS
        assert set(entry["promoted_operations"]) == PROMOTED_RUST_OPERATIONS
        assert entry["dispatch_mode"] in {"rust_native", "python_bridge_visible"}
        if binding == "rust":
            assert entry["dispatch_mode"] == "rust_native"
        else:
            assert entry["dispatch_mode"] == "python_bridge_visible"
