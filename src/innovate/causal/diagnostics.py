"""Diagnostic tools for causal analysis.

This module provides diagnostics for:
- Uncertainty quantification
- Covariate balance checks
- Assumption validation
- Diagnostic warnings and notes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class UncertaintyMetadata:
    """Metadata for uncertainty quantification.

    Attributes
    ----------
        estimand: Type of estimand (ATE, ATT, CATE, etc.)
        point_estimate: Point estimate of effect
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        method: Method used for CI (bootstrap, analytical, etc.)
        n_bootstrap: Number of bootstrap samples (if applicable)
    """

    estimand: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    method: str = "bootstrap"
    n_bootstrap: int | None = None

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def se(self) -> float:
        """Approximate standard error from CI."""
        # Using 95% CI: 1.96 * SE = (upper - lower) / 2
        return self.ci_width / (2 * 1.96)

    def is_significant_at(self, alpha: float = 0.05) -> bool:
        """Check if effect is significantly different from zero at alpha level."""
        return not (self.ci_lower <= 0 <= self.ci_upper)


@dataclass
class DiagnosticsSummary:
    """Collect diagnostic warnings and notes for analysis.

    Attributes
    ----------
        warnings: Dictionary of warning type to message
        notes: Dictionary of note type to message
    """

    warnings: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def add_warning(self, warning_type: str, message: str) -> None:
        """Add a diagnostic warning."""
        self.warnings[warning_type] = message

    def add_note(self, note_type: str, message: str) -> None:
        """Add a diagnostic note."""
        self.notes[note_type] = message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warnings": self.warnings,
            "notes": self.notes,
            "n_warnings": len(self.warnings),
            "n_notes": len(self.notes),
        }


@dataclass
class CovariateBalance:
    """Assess covariate balance between treated and control groups.

    Attributes
    ----------
        treated_covariates: Dictionary of covariate names to arrays for treated
        control_covariates: Dictionary of covariate names to arrays for control
    """

    treated_covariates: dict[str, np.ndarray]
    control_covariates: dict[str, np.ndarray]

    def calculate_smd(self) -> dict[str, float]:
        """Calculate standardized mean differences for all covariates.

        Returns
        -------
            Dictionary of covariate names to SMD values (0-1 scale)
        """
        smd_dict = {}

        for covariate_name in self.treated_covariates:
            if covariate_name not in self.control_covariates:
                continue

            treated = self.treated_covariates[covariate_name]
            control = self.control_covariates[covariate_name]

            # Calculate means
            mean_treated = np.mean(treated)
            mean_control = np.mean(control)

            # Calculate pooled standard deviation
            var_treated = np.var(treated, ddof=1)
            var_control = np.var(control, ddof=1)
            n_treated = len(treated)
            n_control = len(control)

            pooled_var = ((n_treated - 1) * var_treated + (n_control - 1) * var_control) / (n_treated + n_control - 2)
            pooled_sd = np.sqrt(pooled_var)

            # Calculate SMD
            if pooled_sd > 0:
                smd = abs(mean_treated - mean_control) / pooled_sd
            else:
                smd = 0.0

            # Normalize to [0, 1] scale
            smd_normalized = min(smd, 1.0)
            smd_dict[covariate_name] = smd_normalized

        return smd_dict

    def is_balanced(self, threshold: float = 0.1) -> bool:
        """Check if all covariates are balanced.

        Args:
            threshold: SMD threshold for balance (typical: 0.1)

        Returns
        -------
            True if all SMDs are below threshold
        """
        smd_dict = self.calculate_smd()
        return all(smd <= threshold for smd in smd_dict.values())

    def balance_summary(self, threshold: float = 0.1) -> dict[str, Any]:
        """Generate balance summary.

        Args:
            threshold: SMD threshold for balance

        Returns
        -------
            Dictionary with balance statistics
        """
        smd_dict = self.calculate_smd()
        balanced_vars = sum(1 for smd in smd_dict.values() if smd <= threshold)

        return {
            "smd_by_covariate": smd_dict,
            "n_balanced": balanced_vars,
            "n_total": len(smd_dict),
            "all_balanced": self.is_balanced(threshold),
            "max_smd": max(smd_dict.values()) if smd_dict else 0.0,
            "mean_smd": np.mean(list(smd_dict.values())) if smd_dict else 0.0,
        }
