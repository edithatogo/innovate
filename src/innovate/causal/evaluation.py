"""Evaluation workflows for causal policy analysis.

This module provides tools for summarizing causal effects through:
- Pre-post difference-in-differences
- Event-study trajectories
- Counterfactual comparisons
- Heterogeneous effects by subgroups
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PrePostSummary:
    """Pre-post difference-in-differences summary.

    Attributes
    ----------
        pre_treated: Pre-intervention outcomes for treated units
        post_treated: Post-intervention outcomes for treated units
        pre_control: Pre-intervention outcomes for control units
        post_control: Post-intervention outcomes for control units
    """

    pre_treated: np.ndarray
    post_treated: np.ndarray
    pre_control: np.ndarray
    post_control: np.ndarray

    def calculate(self) -> dict[str, float]:
        """Calculate difference-in-differences estimate.

        Returns
        -------
            Dictionary with components of the DiD estimate
        """
        # Change in treated units
        treated_effect = float(np.mean(self.post_treated) - np.mean(self.pre_treated))

        # Change in control units (trend)
        control_trend = float(np.mean(self.post_control) - np.mean(self.pre_control))

        # Difference-in-differences
        did = treated_effect - control_trend

        return {
            "treated_effect": treated_effect,
            "control_trend": control_trend,
            "diff_in_diff": did,
            "pre_treated_mean": float(np.mean(self.pre_treated)),
            "post_treated_mean": float(np.mean(self.post_treated)),
            "pre_control_mean": float(np.mean(self.pre_control)),
            "post_control_mean": float(np.mean(self.post_control)),
        }


@dataclass
class EventStudyTrajectory:
    """Event-study style trajectory with relative time coefficients.

    Attributes
    ----------
        periods: Relative time periods (... -2, -1, 0, 1, 2, ...)
        coefficients: Effect coefficients for each period
        standard_errors: Standard errors for coefficients
    """

    periods: np.ndarray
    coefficients: np.ndarray
    standard_errors: np.ndarray

    def summarize(self) -> dict[str, Any]:
        """Summarize event-study results.

        Returns
        -------
            Dictionary with pre-trend and post-effect summaries
        """
        # Identify pre and post event periods
        pre_mask = self.periods < 0
        post_mask = self.periods > 0
        event_period = self.periods == 0

        pre_coefs = self.coefficients[pre_mask]
        post_coefs = self.coefficients[post_mask]

        # Test for parallel trends (pre-treatment should be near zero)
        pre_trend_mean = float(np.mean(pre_coefs))
        pre_trend_se = float(np.mean(self.standard_errors[pre_mask]))

        # Post-treatment effects
        post_effects_mean = float(np.mean(post_coefs))

        return {
            "pre_trend": pre_trend_mean,
            "pre_trend_se": pre_trend_se,
            "post_effects": post_effects_mean,
            "event_period_coef": float(self.coefficients[event_period][0]) if np.any(event_period) else None,
            "n_pre_periods": int(np.sum(pre_mask)),
            "n_post_periods": int(np.sum(post_mask)),
        }


@dataclass
class CounterfactualComparison:
    """Compare actual outcomes to counterfactual scenario.

    Attributes
    ----------
        actual: Observed/actual outcomes
        counterfactual: Counterfactual outcomes (what would have happened)
    """

    actual: np.ndarray
    counterfactual: np.ndarray

    def summarize(self) -> dict[str, float]:
        """Summarize counterfactual comparison.

        Returns
        -------
            Dictionary with effect estimates
        """
        effect = np.mean(self.actual) - np.mean(self.counterfactual)
        actual_mean = np.mean(self.actual)
        counterfactual_mean = np.mean(self.counterfactual)

        if counterfactual_mean != 0:
            percent_change = (effect / counterfactual_mean) * 100
        else:
            percent_change = 0.0

        return {
            "effect": float(effect),
            "actual_mean": float(actual_mean),
            "counterfactual_mean": float(counterfactual_mean),
            "percent_change": float(percent_change),
        }


@dataclass
class HeterogeneousEffectsSummary:
    """Summary of heterogeneous treatment effects by subgroup.

    Attributes
    ----------
        group_labels: Labels for subgroups
        effects: Treatment effects for each group
        standard_errors: Standard errors for effects
    """

    group_labels: list[str]
    effects: np.ndarray
    standard_errors: np.ndarray

    def summarize(self) -> dict[str, Any]:
        """Summarize heterogeneous effects by group.

        Returns
        -------
            Dictionary with group-level effects
        """
        by_group = {}
        for label, effect, se in zip(self.group_labels, self.effects, self.standard_errors):
            by_group[label] = {
                "effect": float(effect),
                "se": float(se),
                "ci_lower": float(effect - 1.96 * se),
                "ci_upper": float(effect + 1.96 * se),
            }

        # Overall summary
        overall_effect = float(np.mean(self.effects))
        max_effect = float(np.max(self.effects))
        min_effect = float(np.min(self.effects))
        effect_range = max_effect - min_effect

        return {
            "by_group": by_group,
            "overall_effect": overall_effect,
            "max_effect": max_effect,
            "min_effect": min_effect,
            "effect_range": effect_range,
            "n_groups": len(self.group_labels),
        }
