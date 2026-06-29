"""Scenario comparison summary functions.

This module provides functions for generating scenario comparison summaries
including ranking, incremental effects, threshold timing, and uncertainty.
"""

from typing import Any

from innovate.scenario.execution import ScenarioExecution


def compute_ranking(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
    metric: str = "final_adoption",
) -> dict[str, Any]:
    """Compute ranking scores for two scenarios.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.
    metric
        Metric to rank by (default: 'final_adoption').

    Returns
    -------
    dict[str, float]
        Ranking scores for baseline and alternative.
    """
    baseline_score = 0.0
    alternative_score = 0.0

    if metric == "final_adoption" and baseline.adoption_curve is not None and alternative.adoption_curve is not None:
        baseline_score = float(baseline.adoption_curve[-1])
        alternative_score = float(alternative.adoption_curve[-1])

    return {
        "baseline_rank": baseline_score,
        "alternative_rank": alternative_score,
        "metric": metric,
    }


def compute_incremental_effect(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
    metric: str = "final_adoption_increase",
) -> dict[str, Any]:
    """Compute incremental effect of alternative scenario.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.
    metric
        Effect metric to compute.

    Returns
    -------
    dict[str, Any]
        Dictionary with absolute and relative increases.
    """
    if metric == "final_adoption_increase":
        if baseline.adoption_curve is None or alternative.adoption_curve is None:
            return {
                "absolute_increase": None,
                "relative_increase_percent": None,
            }

        baseline_final = float(baseline.adoption_curve[-1])
        alternative_final = float(alternative.adoption_curve[-1])
        absolute = alternative_final - baseline_final
        relative = (absolute / baseline_final * 100) if baseline_final > 0 else 0.0

        return {
            "absolute_increase": absolute,
            "relative_increase_percent": relative,
            "baseline_value": baseline_final,
            "alternative_value": alternative_final,
        }

    return {
        "absolute_increase": None,
        "relative_increase_percent": None,
    }


def compute_threshold_timing(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute time to reach adoption threshold.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.
    threshold
        Adoption threshold (as fraction of market size).

    Returns
    -------
    dict[str, Optional[float]]
        Time to threshold for both scenarios.
    """
    baseline_time = None
    alternative_time = None

    # Compute threshold in absolute adoption
    threshold_absolute = threshold * baseline.scenario.market_size

    # Find time to threshold for baseline
    if baseline.adoption_curve is not None and baseline.time_points is not None:
        for t, adoption in zip(baseline.time_points, baseline.adoption_curve):
            if adoption >= threshold_absolute:
                baseline_time = float(t)
                break

    # Find time to threshold for alternative
    if alternative.adoption_curve is not None and alternative.time_points is not None:
        for t, adoption in zip(alternative.time_points, alternative.adoption_curve):
            if adoption >= threshold_absolute:
                alternative_time = float(t)
                break

    return {
        "baseline_time_to_threshold": baseline_time,
        "alternative_time_to_threshold": alternative_time,
        "threshold": threshold,
        "threshold_absolute": threshold_absolute,
    }


def compute_uncertainty(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Compute uncertainty bounds for scenario outcomes.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.
    confidence_level
        Confidence level for bounds (default: 0.95 = 95%).

    Returns
    -------
    dict[str, float]
        Confidence bounds for both scenarios.
    """
    # Compute uncertainty as simple margin around final values
    # In practice, this could be based on diagnostics, parameter uncertainty, etc.
    margin = 1 - confidence_level  # 0.05 for 95% CI

    baseline_final = 0.0
    baseline_lower = 0.0
    baseline_upper = 0.0

    if baseline.adoption_curve is not None:
        baseline_final = float(baseline.adoption_curve[-1])
        uncertainty_margin = baseline_final * margin
        baseline_lower = baseline_final - uncertainty_margin
        baseline_upper = baseline_final + uncertainty_margin

    alternative_final = 0.0
    alternative_lower = 0.0
    alternative_upper = 0.0

    if alternative.adoption_curve is not None:
        alternative_final = float(alternative.adoption_curve[-1])
        uncertainty_margin = alternative_final * margin
        alternative_lower = alternative_final - uncertainty_margin
        alternative_upper = alternative_final + uncertainty_margin

    return {
        "baseline_final_value": baseline_final,
        "baseline_lower_bound": max(0.0, baseline_lower),
        "baseline_upper_bound": baseline_upper,
        "alternative_final_value": alternative_final,
        "alternative_lower_bound": max(0.0, alternative_lower),
        "alternative_upper_bound": alternative_upper,
        "confidence_level": confidence_level,
    }


def summarize_comparison(
    baseline: ScenarioExecution,
    alternative: ScenarioExecution,
) -> dict[str, Any]:
    """Generate comprehensive comparison summary.

    Parameters
    ----------
    baseline
        Baseline scenario execution.
    alternative
        Alternative scenario execution.

    Returns
    -------
    dict[str, Any]
        Comprehensive summary with all metrics.
    """
    return {
        "ranking": compute_ranking(baseline, alternative),
        "incremental_effect": compute_incremental_effect(baseline, alternative),
        "threshold_timing": compute_threshold_timing(baseline, alternative),
        "uncertainty": compute_uncertainty(baseline, alternative),
        "baseline_scenario": baseline.scenario.to_dict(),
        "alternative_scenario": alternative.scenario.to_dict(),
    }
