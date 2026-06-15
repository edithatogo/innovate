"""Tests for advanced runtime accelerator policy evidence."""

from __future__ import annotations

import json
from pathlib import Path

from innovate.advanced_runtime import AdvancedRuntimePolicy, select_advanced_backend

EVIDENCE_PATH = Path("docs/source/_static/advanced_runtime/performance_evidence.json")


def test_accelerator_policy_records_safe_rust_to_numpy_fallback() -> None:
    """Rust-native preferences should fall back safely when unavailable."""
    selection = select_advanced_backend(
        "uncertainty_calibration",
        policy=AdvancedRuntimePolicy(preferred=("rust", "numpy")),
        available_backends={"rust": False, "numpy": True, "jax": False},
    )

    assert selection.to_dict() == {
        "backend": "numpy",
        "requested_backend": "rust",
        "fallback_used": True,
        "reason": "preferred_backend_unavailable",
    }


def test_advanced_runtime_performance_evidence_is_recorded() -> None:
    """Selected advanced workflows should have lightweight smoke evidence."""
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "advanced_runtime_performance_evidence.v1"
    assert payload["policy"] == "prefer proven accelerator, otherwise fail-safe numpy fallback"
    workflows = {entry["workflow"]: entry for entry in payload["evidence"]}
    assert set(workflows) >= {
        "regime_ensemble",
        "policy_scenario",
        "streaming_update",
        "uncertainty_calibration",
    }
    for entry in workflows.values():
        assert entry["backend"] == "numpy"
        assert entry["fallback_safe"] is True
        assert entry["rows"] > 0
        assert entry["max_runtime_ms"] <= 50.0
        assert entry["evidence_type"] == "local_smoke"
