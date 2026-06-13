"""Regression tests for Rust-native model-family and payload-shape coverage."""

from __future__ import annotations

import json
from pathlib import Path

from innovate.kernel import discover_models

MODEL_FAMILY_COVERAGE = Path("docs/source/_static/rust_native_model_family_coverage.json")
PAYLOAD_SHAPE_COVERAGE = Path("docs/source/_static/rust_native_payload_shape_coverage.json")
SLICE_EVIDENCE = Path("docs/source/_static/rust_native_model_family_slice_evidence.json")


def _model_family_coverage() -> dict:
    return json.loads(MODEL_FAMILY_COVERAGE.read_text())


def _payload_shape_coverage() -> dict:
    return json.loads(PAYLOAD_SHAPE_COVERAGE.read_text())


def _slice_evidence() -> dict:
    return json.loads(SLICE_EVIDENCE.read_text())


def test_every_python_registry_model_family_has_ownership_status() -> None:
    """Every discoverable Python model must have explicit Rust ownership status."""
    registry_keys = {record.key for record in discover_models().models}
    coverage = _model_family_coverage()
    covered_keys = {entry["model_key"] for entry in coverage["families"]}
    allowed_statuses = set(coverage["ownership_status_values"])

    assert covered_keys == registry_keys
    for entry in coverage["families"]:
        assert entry["ownership_status"] in allowed_statuses
        assert entry["native_scope"]
        assert entry["boundary"]


def test_python_reference_model_families_are_explicit_boundaries() -> None:
    """Complex Python-owned model families should not look accidentally unowned."""
    coverage = {
        entry["model_key"]: entry for entry in _model_family_coverage()["families"]
    }

    for key in {
        "complementary_goods",
        "hierarchical",
        "latent_process",
        "mixture",
        "network_diffusion",
        "policy_hazard",
        "regime_switching",
    }:
        assert coverage[key]["ownership_status"] == "python_reference_boundary"
        assert coverage[key]["native_scope"] == "none"

    for key in {"composite", "multi_product", "lotka_volterra"}:
        assert coverage[key]["ownership_status"] == "python_bridge_explicit"


def test_every_stable_payload_shape_has_schema_and_ownership_evidence() -> None:
    """Stable payload shapes require schema fixture evidence and ownership status."""
    coverage = _payload_shape_coverage()
    allowed_statuses = set(coverage["status_values"])
    allowed_ownership = set(coverage["ownership_status_values"])

    for entry in coverage["payload_shapes"]:
        assert entry["status"] in allowed_statuses
        assert entry["ownership_status"] in allowed_ownership
        assert entry["scope"]
        if entry["status"] == "stable":
            assert entry["schema_fixture"] != "none"
            assert Path(entry["schema_fixture"]).exists()
            assert entry["ownership_status"] == "rust_native_promoted"


def test_unstable_payload_shapes_are_not_marked_rust_native() -> None:
    """Provisional/internal/reference payloads should stay explicit boundaries."""
    for entry in _payload_shape_coverage()["payload_shapes"]:
        if entry["status"] != "stable":
            assert entry["ownership_status"] in {
                "python_bridge_explicit",
                "python_reference_boundary",
                "promoted_non_python_backend",
            }


def test_promoted_diffusion_and_substitution_families_have_slice_evidence() -> None:
    """Promoted stable families should link to parity and error evidence."""
    evidence = _slice_evidence()
    entries = {
        entry["model_key"]: entry
        for entry in evidence["stable_diffusion_and_substitution"]
    }

    assert set(entries) == {"bass", "logistic", "gompertz", "fisher_pry", "norton_bass"}
    for entry in entries.values():
        assert entry["ownership_status"] == "rust_native_promoted"
        assert entry["promoted_operations"]
        assert "Rust operation tests" in entry["parity_evidence"]
        assert entry["error_mapping_evidence"]
