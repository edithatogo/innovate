"""Tests for advanced modeling runtime contracts."""

from __future__ import annotations

import json

import pytest

from innovate.advanced_runtime import (
    AdvancedCapability,
    AdvancedResult,
    AdvancedRuntimePolicy,
    detect_advanced_backends,
    get_advanced_capability,
    list_advanced_capabilities,
    select_advanced_backend,
)


def test_advanced_capability_registry_marks_stability_and_dependencies() -> None:
    """Advanced workflows expose machine-readable stability and dependency metadata."""
    capabilities = {capability.key: capability for capability in list_advanced_capabilities()}

    assert set(capabilities) >= {
        "regime_ensemble",
        "policy_scenario",
        "streaming_update",
        "uncertainty_calibration",
    }
    assert capabilities["regime_ensemble"] == AdvancedCapability(
        key="regime_ensemble",
        family="ensemble",
        stability="experimental",
        result_schema="advanced_result.v1",
        optional_dependencies=(),
        supported_backends=("numpy", "jax", "rust"),
    )
    assert capabilities["policy_scenario"].stability == "stable"
    assert capabilities["streaming_update"].supports_incremental is True
    assert capabilities["uncertainty_calibration"].supports_uncertainty is True
    assert capabilities["uncertainty_calibration"].to_dict()["supported_backends"] == [
        "numpy",
        "jax",
        "rust",
    ]


def test_advanced_capability_validation_and_lookup_errors() -> None:
    """Capability metadata should reject invalid definitions and unknown keys."""
    with pytest.raises(ValueError, match="key"):
        AdvancedCapability(key="", family="ensemble", stability="experimental")
    with pytest.raises(ValueError, match="family"):
        AdvancedCapability(key="candidate", family="", stability="experimental")
    with pytest.raises(KeyError, match="unknown"):
        get_advanced_capability("unknown")


def test_advanced_result_serializes_to_stable_json_payload() -> None:
    """Result objects should round-trip through JSON without losing metadata."""
    result = AdvancedResult(
        workflow="policy_scenario",
        stability="stable",
        backend="numpy",
        time=[1.0, 2.0],
        mean=[10.0, 12.5],
        metadata={"scenario": "rebate"},
        diagnostics={"coverage": 0.9},
        lower=[9.0, 11.0],
        upper=[11.0, 14.0],
    )

    payload = result.to_dict()
    assert payload["schema_version"] == "advanced_result.v1"
    assert payload["capability"] == {
        "workflow": "policy_scenario",
        "stability": "stable",
        "backend": "numpy",
    }
    assert payload["time"] == [1.0, 2.0]
    assert payload["mean"] == [10.0, 12.5]
    assert payload["lower"] == [9.0, 11.0]
    assert payload["upper"] == [11.0, 14.0]
    assert json.loads(json.dumps(payload)) == payload


def test_advanced_result_rejects_unknown_stability() -> None:
    """Advanced result metadata should fail fast on invalid stability labels."""
    with pytest.raises(ValueError, match="stability"):
        AdvancedResult(workflow="policy_scenario", stability="private", backend="numpy")
    with pytest.raises(ValueError, match="workflow"):
        AdvancedResult(workflow="", stability="stable", backend="numpy")
    with pytest.raises(ValueError, match="backend"):
        AdvancedResult(workflow="policy_scenario", stability="stable", backend="")


def test_runtime_policy_requires_a_preferred_backend() -> None:
    """Execution policy should fail fast when no backend preference is declared."""
    with pytest.raises(ValueError, match="preferred backend"):
        AdvancedRuntimePolicy(preferred=())


def test_select_advanced_backend_falls_back_when_optional_backend_missing() -> None:
    """Accelerator policy should be explicit about safe fallback behavior."""
    policy = AdvancedRuntimePolicy(preferred=("jax", "numpy"), allow_fallback=True)

    selection = select_advanced_backend(
        "policy_scenario",
        policy=policy,
        available_backends={"jax": False, "numpy": True},
    )

    assert selection.backend == "numpy"
    assert selection.requested_backend == "jax"
    assert selection.fallback_used is True
    assert selection.reason == "preferred_backend_unavailable"
    assert selection.to_dict()["fallback_used"] is True


def test_select_advanced_backend_uses_first_available_preferred_backend() -> None:
    """Policy resolution should preserve preferred ordering when a backend is available."""
    selection = select_advanced_backend(
        "policy_scenario",
        policy=AdvancedRuntimePolicy(preferred=("numpy", "jax"), allow_fallback=True),
        available_backends={"numpy": True, "jax": True},
    )

    assert selection.backend == "numpy"
    assert selection.fallback_used is False
    assert selection.reason == "preferred_backend_available"


def test_select_advanced_backend_can_fail_closed() -> None:
    """Strict execution should fail instead of silently changing backend."""
    policy = AdvancedRuntimePolicy(preferred=("rust",), allow_fallback=False)

    with pytest.raises(RuntimeError, match="rust"):
        select_advanced_backend(
            "regime_ensemble",
            policy=policy,
            available_backends={"rust": False, "numpy": True},
        )


def test_select_advanced_backend_reports_no_available_backend() -> None:
    """Runtime selection should fail clearly when every supported backend is absent."""
    with pytest.raises(RuntimeError, match="No available backend"):
        select_advanced_backend(
            "policy_scenario",
            policy=AdvancedRuntimePolicy(preferred=("jax",), allow_fallback=True),
            available_backends={"jax": False, "numpy": False, "rust": False},
        )


def test_detect_advanced_backends_reports_numpy_available() -> None:
    """Backend detection should always include the dependency-free NumPy path."""
    detected = detect_advanced_backends()

    assert detected["numpy"] is True
    assert set(detected) == {"numpy", "jax", "rust"}


def test_list_advanced_capabilities_deterministic_order() -> None:
    """list_advanced_capabilities should return capabilities as a tuple sorted by key."""
    capabilities = list_advanced_capabilities()

    assert isinstance(capabilities, tuple)
    assert len(capabilities) > 0
    assert all(isinstance(c, AdvancedCapability) for c in capabilities)

    keys = [c.key for c in capabilities]
    assert keys == sorted(keys), "Capabilities must be returned in deterministic (sorted) order"

    # Optional but good: Verify it returns the full known set
    expected_keys = {
        "regime_ensemble",
        "policy_scenario",
        "streaming_update",
        "uncertainty_calibration",
    }
    assert set(keys) == expected_keys
