"""Causal inference tools for policy evaluation and impact analysis."""

from __future__ import annotations

from innovate.causal.counterfactual import CounterfactualAnalysis
from innovate.causal.diagnostics import (
    CovariateBalance,
    DiagnosticsSummary,
    UncertaintyMetadata,
)
from innovate.causal.evaluation import (
    CounterfactualComparison,
    EventStudyTrajectory,
    HeterogeneousEffectsSummary,
    PrePostSummary,
)
from innovate.causal.policy import (
    CausalModel,
    CausalModelContract,
    InterventionContract,
    PolicyEvaluationError,
    TreatmentEffectEstimator,
)
from innovate.causal.sensitivity import (
    EValue,
    RosenbaumBounds,
    SensitivityAnalysis,
)

__all__ = [
    "CounterfactualAnalysis",
    "CausalModel",
    "CausalModelContract",
    "InterventionContract",
    "PolicyEvaluationError",
    "TreatmentEffectEstimator",
    "RosenbaumBounds",
    "EValue",
    "SensitivityAnalysis",
    "PrePostSummary",
    "EventStudyTrajectory",
    "CounterfactualComparison",
    "HeterogeneousEffectsSummary",
    "UncertaintyMetadata",
    "DiagnosticsSummary",
    "CovariateBalance",
]
