"""End-to-end advanced runtime workflow example."""

from __future__ import annotations

import numpy as np

from innovate.advanced_runtime import (
    StreamingUpdateConfig,
    calibrate_prediction_intervals,
    compare_policy_scenarios,
    compose_regime_ensemble,
    update_streaming_forecast,
)


def build_report() -> dict[str, dict[str, object]]:
    """Build a JSON-friendly report for every advanced runtime workflow."""
    time = [1, 2, 3, 4, 5, 6, 7, 8]
    observed = np.array([4.0, 8.2, 14.1, 23.4, 37.5, 54.7, 70.8, 82.0])
    slow = observed * 0.94
    fast = observed * 1.04
    ensemble = compose_regime_ensemble(
        time=time,
        predictions={"slow": slow, "fast": fast},
        observed=observed,
        weights={"slow": 0.45, "fast": 0.55},
        assumptions=["Two plausible adoption regimes are combined."],
    )

    policy_observed = np.array([3.0, 6.0, 10.0, 17.0, 29.0, 45.0, 61.0, 75.0])
    baseline = policy_observed * np.array([1.0, 1.0, 1.0, 0.93, 0.9, 0.88, 0.86, 0.84])
    policy = compare_policy_scenarios(
        time=time,
        baseline=baseline,
        intervention=policy_observed,
        observed=policy_observed,
        scenario_name="rebate",
        assumptions=["A rebate policy starts at period 4."],
        covariates={"rebate_active": [0, 0, 0, 1, 1, 1, 1, 1]},
    )

    config = StreamingUpdateConfig(
        previous_time=[1, 2, 3, 4],
        previous_observed=[5.0, 9.0, 15.0, 24.0],
        new_time=[5, 6],
        new_observed=[36.0, 49.0],
        assumptions=["Periods 5 and 6 arrive after the initial fit window."],
    )
    streaming = update_streaming_forecast(config)

    calibration_observed = np.array([2.0, 5.0, 9.0, 16.0, 26.0, 39.0, 53.0])
    predicted = calibration_observed * np.array([1.03, 0.98, 1.02, 0.97, 0.95, 1.03, 0.98])
    calibration = calibrate_prediction_intervals(
        time=[1, 2, 3, 4, 5, 6, 7],
        observed=calibration_observed,
        predicted=predicted,
        confidence=0.8,
        holdout=[0, 0, 0, 0, 1, 1, 1],
        assumptions=["Periods 5 through 7 are holdout coverage checks."],
    )

    return {
        "ensemble": ensemble.to_dict(),
        "policy": policy.to_dict(),
        "streaming": streaming.to_dict(),
        "calibration": calibration.to_dict(),
    }


if __name__ == "__main__":
    report = build_report()
    for name, payload in report.items():
        print(name, payload["capability"])
