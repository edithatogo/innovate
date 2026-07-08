"""Integration coverage for advanced modeling workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from innovate.advanced_runtime import (
    CalibrationConfig,
    calibrate_prediction_intervals,
    compare_policy_scenarios,
    compose_regime_ensemble,
    update_streaming_forecast,
)

FIXTURE_PATH = Path("tests/fixtures/advanced_runtime/workflows.json")


def _case(workflow: str) -> dict[str, object]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["workflow"] == workflow:
            return case
    raise AssertionError(f"Missing fixture case for {workflow}")


def test_compose_regime_ensemble_scores_and_serializes_result() -> None:
    """Regime ensembles should combine compatible trajectories with scores."""
    fixture = _case("regime_ensemble")
    time = fixture["time"]
    observed = np.asarray(fixture["observed"], dtype=float)
    slow_regime = observed * 0.94
    fast_regime = observed * 1.04

    result = compose_regime_ensemble(
        time=time,
        predictions={"slow": slow_regime, "fast": fast_regime},
        observed=observed,
        weights={"slow": 0.45, "fast": 0.55},
        assumptions=fixture["assumptions"],
    )

    payload = result.to_dict()
    assert result.workflow == "regime_ensemble"
    assert result.stability == "experimental"
    assert result.backend == "numpy"
    assert payload["mean"] == pytest.approx((slow_regime * 0.45 + fast_regime * 0.55).tolist())
    assert payload["metadata"]["weights"] == {"fast": 0.55, "slow": 0.45}
    assert payload["metadata"]["assumptions"] == fixture["assumptions"]
    assert payload["diagnostics"]["mae"] < 3.0
    assert payload["diagnostics"]["rmse"] < 4.0


def test_compare_policy_scenarios_reports_auditable_effect_summary() -> None:
    """Policy scenarios should compare intervention and baseline trajectories."""
    fixture = _case("policy_scenario")
    observed = np.asarray(fixture["observed"], dtype=float)
    baseline = observed * np.array([1.0, 1.0, 1.0, 0.93, 0.9, 0.88, 0.86, 0.84])
    intervention = observed

    result = compare_policy_scenarios(
        time=fixture["time"],
        baseline=baseline,
        intervention=intervention,
        observed=observed,
        scenario_name="rebate",
        assumptions=fixture["assumptions"],
        covariates=fixture["covariates"],
    )

    payload = result.to_dict()
    assert result.workflow == "policy_scenario"
    assert result.stability == "stable"
    assert payload["mean"] == pytest.approx(intervention.tolist())
    assert payload["metadata"]["scenario_name"] == "rebate"
    assert payload["metadata"]["incremental_effect"] == pytest.approx(float(np.sum(intervention - baseline)))
    assert payload["metadata"]["relative_lift_final"] == pytest.approx(intervention[-1] / baseline[-1] - 1.0)
    assert payload["metadata"]["covariates"]["rebate_active"] == fixture["covariates"]["rebate_active"]
    assert payload["diagnostics"]["baseline_mae"] > payload["diagnostics"]["intervention_mae"]


def test_update_streaming_forecast_returns_incremental_state_and_diagnostics() -> None:
    """Streaming updates should append new observations without losing state metadata."""
    fixture = _case("streaming_update")
    initial_time = fixture["time"][:4]
    initial_observed = fixture["observed"][:4]
    new_time = fixture["time"][4:]
    new_observed = fixture["observed"][4:]

    result = update_streaming_forecast(
        previous_time=initial_time,
        previous_observed=initial_observed,
        new_time=new_time,
        new_observed=new_observed,
        assumptions=fixture["assumptions"],
    )

    payload = result.to_dict()
    assert result.workflow == "streaming_update"
    assert result.stability == "experimental"
    assert payload["time"] == fixture["time"]
    assert payload["mean"] == fixture["observed"]
    assert payload["metadata"]["previous_count"] == 4
    assert payload["metadata"]["new_count"] == 2
    assert payload["metadata"]["state"]["last_observed"] == fixture["observed"][-1]
    assert payload["diagnostics"]["incremental_growth"] == pytest.approx(25.0)


def test_calibrate_prediction_intervals_reports_coverage_and_residuals() -> None:
    """Calibration should produce intervals, residual diagnostics, and coverage metadata."""
    fixture = _case("uncertainty_calibration")
    observed = np.asarray(fixture["observed"], dtype=float)
    predicted = observed * np.array([1.03, 0.98, 1.02, 0.97, 0.95, 1.03, 0.98])

    config = CalibrationConfig(
        confidence=0.8,
        holdout=fixture["covariates"]["holdout"],
        assumptions=fixture["assumptions"],
    )
    result = calibrate_prediction_intervals(
        time=fixture["time"],
        observed=observed,
        predicted=predicted,
        config=config,
    )

    payload = result.to_dict()
    assert result.workflow == "uncertainty_calibration"
    assert result.stability == "stable"
    assert len(payload["lower"]) == len(fixture["time"])
    assert len(payload["upper"]) == len(fixture["time"])
    assert payload["diagnostics"]["coverage"] >= 0.8
    assert payload["diagnostics"]["holdout_coverage"] >= 0.8
    assert payload["diagnostics"]["residual_mean"] == pytest.approx(float(np.mean(observed - predicted)))
    assert payload["metadata"]["confidence"] == 0.8
