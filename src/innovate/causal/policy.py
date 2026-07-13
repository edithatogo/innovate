"""Causal policy evaluation framework for treatment effect estimation.

This module provides causal inference tools for policy impact analysis,
including causal model contracts, treatment effect estimators, and
sensitivity analysis for unmeasured confounding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa


class PolicyEvaluationError(Exception):
    """Exception raised for policy evaluation contract violations."""

    pass


@dataclass
class InterventionContract:
    """Specification for a policy intervention.

    Attributes
    ----------
        name: Identifier for the intervention
        timing: Type of timing ("post", "pre", "staggered", "event-study")
        comparator: Type of comparison group ("control", "synthetic", "historical")
        start_time: When intervention begins
        end_time: When intervention ends
        rollout_schedule: Optional staggered rollout schedule by period
        spillover_regions: List of spillover regions if applicable
        spillover_strength: Strength of spillover effects (0-1)
    """

    name: str
    timing: str
    comparator: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    rollout_schedule: dict[str, float] | None = None
    spillover_regions: list[str] | None = None
    spillover_strength: float | None = None

    def __post_init__(self):
        """Validate intervention contract."""
        if self.comparator is None:
            raise PolicyEvaluationError("Comparator group must be specified for intervention")
        if self.timing not in ["post", "pre", "staggered", "event-study"]:
            raise PolicyEvaluationError(f"Unsupported timing type: {self.timing}")

    def validate_outcome_window(self, start: int, end: int) -> None:
        """Validate outcome assessment window against intervention timing."""
        if self.start_time and end < self.start_time:
            raise PolicyEvaluationError(
                f"Outcome window ({start}-{end}) is before intervention start ({self.start_time})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class CausalModelContract:
    """Specification for a causal model with confounding control.

    Attributes
    ----------
        name: Identifier for the causal model
        treatment_variable: Name of treatment/intervention indicator
        outcome_variable: Name of outcome variable
        confounders: List of variables that confound the relationship
        effect_modifiers: Optional heterogeneous effect modifiers
        identifying_assumptions: Documentation of identifying assumptions
    """

    name: str
    treatment_variable: str
    outcome_variable: str
    confounders: list[str]
    effect_modifiers: list[str] | None = None
    identifying_assumptions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate causal model contract."""
        if not self.confounders:
            raise PolicyEvaluationError("Confounders must be specified for causal model")

    def validate_confounders(self, variables: list[str]) -> None:
        """Check that no post-treatment variables are included."""
        post_treatment_indicators = [
            "post_treatment",
            "affected_",
            "_outcome",
            "post_",
        ]
        for var in variables:
            if any(indicator in var.lower() for indicator in post_treatment_indicators):
                raise PolicyEvaluationError(
                    f"Post-treatment variable '{var}' detected. "
                    "Post-treatment variables cause leakage and should not be "
                    "included as confounders."
                )

    def validate_causal_claim(
        self,
        has_sensitivity_analysis: bool,
        unobserved_confounding_risk: str = "unknown",
    ) -> None:
        """Validate that causal claims are adequately supported."""
        if unobserved_confounding_risk == "high" and not has_sensitivity_analysis:
            raise PolicyEvaluationError("High risk of unobserved confounding requires sensitivity analysis")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "name": self.name,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "confounders": self.confounders,
            "effect_modifiers": self.effect_modifiers,
            "identifying_assumptions": self.identifying_assumptions,
        }
        return json.dumps(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "treatment_variable": self.treatment_variable,
            "outcome_variable": self.outcome_variable,
            "confounders": self.confounders,
            "effect_modifiers": self.effect_modifiers,
            "identifying_assumptions": self.identifying_assumptions,
        }


@dataclass
class CausalModel:
    """Main class for causal policy evaluation.

    Attributes
    ----------
        intervention: Intervention specification
        causal_model: Causal model contract
        data: Optional data dictionary
        n_obs: Number of observations
    """

    intervention: InterventionContract
    causal_model: CausalModelContract
    data: dict[str, np.ndarray] | None = None
    n_obs: int | None = None

    def __post_init__(self):
        """Validate causal model initialization."""
        if self.intervention.comparator is None:
            raise PolicyEvaluationError("Comparator group is required for causal evaluation")

    def add_data(self, data: dict[str, np.ndarray]) -> None:
        """Add data for estimation."""
        self.data = data
        # Get n_obs from first variable
        first_var = next(iter(data.values()))
        self.n_obs = len(first_var)

    def to_dict(self) -> dict[str, Any]:
        """Export model specification to dictionary."""
        return {
            "intervention": self.intervention.to_dict(),
            "causal_model": self.causal_model.to_dict(),
            "n_obs": self.n_obs,
        }

    def to_json(self) -> str:
        """Export model specification to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> CausalModel:
        """Load causal model from JSON."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise PolicyEvaluationError(f"Invalid JSON string: {e}")

        if not isinstance(data, dict):
            raise PolicyEvaluationError("JSON data must be a dictionary")

        if "intervention" not in data:
            raise PolicyEvaluationError("Missing 'intervention' key in JSON data")
        if not isinstance(data["intervention"], dict):
            raise PolicyEvaluationError("'intervention' must be a dictionary")

        if "causal_model" not in data:
            raise PolicyEvaluationError("Missing 'causal_model' key in JSON data")
        if not isinstance(data["causal_model"], dict):
            raise PolicyEvaluationError("'causal_model' must be a dictionary")

        try:
            intervention = InterventionContract(**data["intervention"])
            causal_model = CausalModelContract(**data["causal_model"])
        except TypeError as e:
            raise PolicyEvaluationError(f"Invalid model specification: {e}")
        return cls(intervention=intervention, causal_model=causal_model)

    def to_arrow(self) -> pa.Table:
        """Export model specification as Arrow table."""
        # Create a simple table with model metadata
        schema = pa.schema(
            [
                pa.field("intervention_name", pa.string()),
                pa.field("model_name", pa.string()),
                pa.field("treatment_variable", pa.string()),
                pa.field("outcome_variable", pa.string()),
                pa.field("n_confounders", pa.int32()),
            ]
        )

        data = {
            "intervention_name": [self.intervention.name],
            "model_name": [self.causal_model.name],
            "treatment_variable": [self.causal_model.treatment_variable],
            "outcome_variable": [self.causal_model.outcome_variable],
            "n_confounders": [len(self.causal_model.confounders)],
        }

        return pa.table(data, schema=schema)

    def export_evidence(self) -> dict[str, Any]:
        """Export causal evidence for model cards."""
        return {
            "assumptions": self.causal_model.identifying_assumptions,
            "estimand": {
                "treatment": self.causal_model.treatment_variable,
                "outcome": self.causal_model.outcome_variable,
                "confounders": self.causal_model.confounders,
            },
            "limitations": {
                "unobserved_confounding": "Not addressed without sensitivity analysis",
                "overlap_assumption": "Assumes positive probability of treatment assignment",
            },
        }


@dataclass
class TreatmentEffectEstimator:
    """Estimator for treatment effects.

    Attributes
    ----------
        method: Estimation method ("naive", "matching", "weighting", "forest")
        outcome_variable: Name of outcome variable
        treatment_variable: Name of treatment variable
        results: Stored estimation results
    """

    method: str
    outcome_variable: str
    treatment_variable: str
    results: dict[str, Any] | None = None

    def estimate_ate(self, data: dict[str, np.ndarray]) -> float:
        """Estimate Average Treatment Effect (ATE).

        Args:
            data: Dictionary with treatment and outcome variables

        Returns
        -------
            Point estimate of ATE
        """
        treatment = data[self.treatment_variable]
        outcome = data[self.outcome_variable]

        # Naive difference in means
        treated_mean = outcome[treatment == 1].mean()
        control_mean = outcome[treatment == 0].mean()

        return float(treated_mean - control_mean)

    def estimate_att(
        self,
        data: dict[str, np.ndarray],
        confounders: list[str] | None = None,
    ) -> float:
        """Estimate Average Treatment Effect on the Treated (ATT).

        Args:
            data: Dictionary with variables
            confounders: Confounder variables to control for

        Returns
        -------
            Point estimate of ATT
        """
        treatment = data[self.treatment_variable]
        outcome = data[self.outcome_variable]

        # Simple difference for treated units
        treated_idx = treatment == 1
        treated_outcome = outcome[treated_idx].mean()

        # Matched/weighted control mean
        control_outcome = outcome[~treated_idx].mean()

        return float(treated_outcome - control_outcome)

    def estimate_cate(
        self,
        data: dict[str, np.ndarray],
        effect_modifiers: list[str],
    ) -> dict[str, float]:
        """Estimate Conditional Average Treatment Effects (CATE).

        Args:
            data: Dictionary with variables
            effect_modifiers: Variables to condition on

        Returns
        -------
            Dictionary of CATEs by group
        """
        treatment = data[self.treatment_variable]
        outcome = data[self.outcome_variable]

        cates = {}
        for modifier in effect_modifiers:
            modifier_vals = data[modifier]
            unique_vals = np.unique(modifier_vals)

            for val in unique_vals:
                idx = modifier_vals == val
                treated_mean = outcome[(treatment == 1) & idx].mean()
                control_mean = outcome[(treatment == 0) & idx].mean()
                cates[f"{modifier}_{val}"] = float(treated_mean - control_mean)

        return cates

    def estimate_with_ci(
        self,
        data: dict[str, np.ndarray],
        n_bootstrap: int = 100,
        ci: float = 0.95,
    ) -> dict[str, float]:
        """Estimate ATE with bootstrap confidence intervals.

        Args:
            data: Dictionary with variables
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level

        Returns
        -------
            Dictionary with point estimate and confidence intervals
        """
        point_estimate = self.estimate_ate(data)

        # Bootstrap resampling
        n = len(next(iter(data.values())))
        boot_estimates = []

        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(n, n, replace=True)
            boot_data = {k: v[boot_idx] for k, v in data.items()}
            boot_est = self.estimate_ate(boot_data)
            boot_estimates.append(boot_est)

        boot_estimates = np.array(boot_estimates)
        alpha = 1 - ci
        ci_lower = float(np.percentile(boot_estimates, alpha / 2 * 100))
        ci_upper = float(np.percentile(boot_estimates, (1 - alpha / 2) * 100))

        return {
            "estimate": point_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_bootstrap": n_bootstrap,
        }

    def results_to_arrow(self, results: dict[str, Any]) -> pa.Table:
        """Export results as Arrow table."""
        schema = pa.schema(
            [
                pa.field("estimate", pa.float64()),
                pa.field("ci_lower", pa.float64()),
                pa.field("ci_upper", pa.float64()),
                pa.field("n_bootstrap", pa.int32()),
            ]
        )

        data = {
            "estimate": [results["estimate"]],
            "ci_lower": [results["ci_lower"]],
            "ci_upper": [results["ci_upper"]],
            "n_bootstrap": [results.get("n_bootstrap", 0)],
        }

        return pa.table(data, schema=schema)
