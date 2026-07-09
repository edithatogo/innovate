"""Sensitivity analysis for unmeasured confounding in causal inference.

This module provides tools to assess robustness of causal estimates to
violations of the unconfoundedness assumption, including Rosenbaum bounds,
E-values, and other sensitivity parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RosenbaumBounds:
    """Calculate Rosenbaum bounds for sensitivity to hidden bias.

    Attributes
    ----------
        matched_pairs: Number of matched pairs
        treated_outcomes: Outcomes for treated units
        control_outcomes: Outcomes for control units
    """

    matched_pairs: int
    treated_outcomes: list[float]
    control_outcomes: list[float]

    def calculate(self, gamma: float) -> tuple[float, float]:
        """Calculate bounds for a given gamma (odds of differential assignment).

        Args:
            gamma: Odds of treatment assignment odds ratio (1 = no hidden bias)

        Returns
        -------
            Tuple of (lower_bound, upper_bound)
        """
        # Simplified Rosenbaum bounds calculation
        # As gamma increases, bounds widen to reflect uncertainty

        treated = np.array(self.treated_outcomes)
        control = np.array(self.control_outcomes)

        if len(treated) != len(control):
            # Handle mismatched lengths by truncating
            min_len = min(len(treated), len(control))
            treated = treated[:min_len]
            control = control[:min_len]

        diff = treated - control

        # Point estimate
        point = np.mean(diff)

        # Standard error
        se = np.std(diff) / np.sqrt(len(diff))

        # Bounds widen with gamma
        # When gamma = 1, bounds collapse to point estimate
        # As gamma increases, bounds widen
        margin = np.log(gamma) * se * 2

        lower = point - margin
        upper = point + margin

        return (lower, upper)


@dataclass
class EValue:
    """E-value for robustness to unmeasured confounding.

    Attributes
    ----------
        point_estimate: Point estimate of effect
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float

    def calculate(self) -> float:
        """Calculate E-value.

        Returns
        -------
            E-value indicating strength of unmeasured confounder
        """
        # E-value is the minimum bias factor required to change inference
        # For a point estimate: max(effect, 1/effect)

        if self.point_estimate >= 1:
            return self.point_estimate + np.sqrt(self.point_estimate * (self.point_estimate - 1))
        recip = 1 / self.point_estimate
        return recip + np.sqrt(recip * (recip - 1))


@dataclass
class SensitivityAnalysis:
    """Conduct sensitivity analysis for unmeasured confounding.

    Attributes
    ----------
        point_estimate: Effect estimate
        method: Method for sensitivity ("rosenbaum", "e_value", "bounds")
    """

    point_estimate: float
    method: str = "rosenbaum"

    def analyze(self, gamma_range: list[float]) -> list[dict[str, Any]]:
        """Analyze sensitivity across a range of gamma values.

        Args:
            gamma_range: List of gamma values to evaluate

        Returns
        -------
            List of sensitivity results
        """
        results = []
        for gamma in gamma_range:
            result = {
                "gamma": gamma,
                "robustness": self._calculate_robustness(gamma),
            }
            results.append(result)
        return results

    def analyze_scenario(
        self,
        gamma: float,
        direction: str = "both",
    ) -> dict[str, float]:
        """Analyze sensitivity for a specific unmeasured confounding scenario.

        Args:
            gamma: Odds ratio for unmeasured confounding
            direction: Direction of bias ("both", "positive", "negative")

        Returns
        -------
            Bounds under the scenario
        """
        robustness = self._calculate_robustness(gamma)

        if direction == "both":
            lower = self.point_estimate - robustness
            upper = self.point_estimate + robustness
        elif direction == "positive":
            lower = self.point_estimate
            upper = self.point_estimate + robustness
        else:  # negative
            lower = self.point_estimate - robustness
            upper = self.point_estimate

        return {
            "gamma": gamma,
            "point_estimate": self.point_estimate,
            "lower_bound": lower,
            "upper_bound": upper,
            "robustness_margin": robustness,
        }

    def _calculate_robustness(self, gamma: float) -> float:
        """Calculate robustness margin for given gamma."""
        # Simplified: larger gamma means wider bounds
        base_robustness = abs(self.point_estimate) * 0.1
        margin = base_robustness * (gamma - 1)
        return margin
