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
