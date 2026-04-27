"""Shared diagnostics and uncertainty contract for model fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters.residual_analysis import ResidualAnalysis, analyze_residuals

WarningSeverity = Literal["info", "warning", "error"]
ReportType = Literal["point_estimate", "bootstrap_interval", "posterior_summary", "unsupported"]
SupportLevel = Literal["supported", "partial", "unsupported"]
Provenance = Literal["deterministic", "bootstrap", "bayesian", "unknown"]


@dataclass(frozen=True)
class DiagnosticsWarning:
    """Structured warning emitted by diagnostics and uncertainty helpers."""

    code: str
    message: str
    severity: WarningSeverity = "warning"

    def to_dict(self) -> dict[str, str]:
        """Serialize the warning to a JSON-friendly dictionary."""
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass(frozen=True)
class UncertaintySummary:
    """Canonical uncertainty summary for deterministic and probabilistic fitters."""

    report_type: ReportType
    provenance: Provenance
    support_level: SupportLevel
    level: float | None = None
    lower: dict[str, float] = field(default_factory=dict)
    upper: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    samples: dict[str, np.ndarray] = field(default_factory=dict)
    note: str = ""

    @classmethod
    def point_estimate(cls, provenance: Provenance = "deterministic", note: str = "") -> UncertaintySummary:
        """Create a summary for deterministic point estimates."""
        return cls(
            report_type="point_estimate",
            provenance=provenance,
            support_level="supported",
            note=note,
        )

    @classmethod
    def bootstrap_interval(
        cls,
        lower: dict[str, float],
        upper: dict[str, float],
        median: dict[str, float] | None = None,
        *,
        level: float = 0.95,
        note: str = "",
    ) -> UncertaintySummary:
        """Create a bootstrap interval summary."""
        return cls(
            report_type="bootstrap_interval",
            provenance="bootstrap",
            support_level="supported",
            level=level,
            lower=lower,
            upper=upper,
            median={} if median is None else median,
            note=note,
        )

    @classmethod
    def posterior_summary(
        cls,
        lower: dict[str, float],
        upper: dict[str, float],
        median: dict[str, float] | None = None,
        *,
        level: float = 0.95,
        samples: dict[str, np.ndarray] | None = None,
        note: str = "",
    ) -> UncertaintySummary:
        """Create a Bayesian posterior summary."""
        return cls(
            report_type="posterior_summary",
            provenance="bayesian",
            support_level="supported",
            level=level,
            lower=lower,
            upper=upper,
            median={} if median is None else median,
            samples={} if samples is None else samples,
            note=note,
        )

    @classmethod
    def unsupported(cls, note: str, provenance: Provenance = "unknown") -> UncertaintySummary:
        """Create an explicit unsupported uncertainty marker."""
        return cls(
            report_type="unsupported",
            provenance=provenance,
            support_level="unsupported",
            note=note,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the uncertainty summary to a dictionary."""
        return {
            "report_type": self.report_type,
            "provenance": self.provenance,
            "support_level": self.support_level,
            "level": self.level,
            "lower": self.lower,
            "upper": self.upper,
            "median": self.median,
            "samples": {name: sample.tolist() for name, sample in self.samples.items()},
            "note": self.note,
        }


@dataclass
class DiagnosticsContract:
    """Canonical diagnostics surface for a fitted model."""

    metrics: dict[str, float] = field(default_factory=dict)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    residual_analysis: ResidualAnalysis | None = None
    warnings: list[DiagnosticsWarning] = field(default_factory=list)
    uncertainty: UncertaintySummary = field(default_factory=UncertaintySummary.point_estimate)
    support_level: SupportLevel = "supported"
    provenance: Provenance = "unknown"
    comparison_family: str = "deterministic"
    model_name: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the contract into a dictionary for downstream consumers."""
        return {
            "metrics": self.metrics,
            "residuals": self.residuals.tolist(),
            "residual_analysis": None if self.residual_analysis is None else self.residual_analysis.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "uncertainty": self.uncertainty.to_dict(),
            "support_level": self.support_level,
            "provenance": self.provenance,
            "comparison_family": self.comparison_family,
            "model_name": self.model_name,
        }


def build_diagnostics_contract(
    model: DiffusionModel,
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    provenance: Provenance = "deterministic",
    uncertainty: UncertaintySummary | None = None,
    warnings: list[DiagnosticsWarning] | None = None,
    model_name: str = "",
) -> DiagnosticsContract:
    """Build a canonical diagnostics contract for a fitted model.

    Unsupported models are reported explicitly rather than failing silently.
    """
    warning_list = list(warnings or [])

    if not hasattr(model, "predict") or not callable(model.predict):
        warning_list.append(
            DiagnosticsWarning(
                code="model_unavailable",
                message="Model does not expose a callable predict method.",
            ),
        )
        return DiagnosticsContract(
            uncertainty=uncertainty
            or UncertaintySummary.unsupported(
                "Model does not expose a callable predict method.",
                provenance="unknown",
            ),
            warnings=warning_list,
            support_level="unsupported",
            provenance="unknown",
            comparison_family="unsupported",
            model_name=model_name,
        )

    if not getattr(model, "params_", None):
        warning_list.append(
            DiagnosticsWarning(
                code="model_unfitted",
                message="Model has not been fitted yet.",
            ),
        )
        return DiagnosticsContract(
            uncertainty=uncertainty
            or UncertaintySummary.unsupported(
                "Model has not been fitted yet.",
                provenance=provenance,
            ),
            warnings=warning_list,
            support_level="unsupported",
            provenance=provenance,
            comparison_family="unsupported",
            model_name=model_name,
        )

    from innovate.utils.model_evaluation import get_fit_metrics  # Local import avoids cycle.

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    y_pred = np.asarray(model.predict(t_arr), dtype=float)
    residuals = y_arr - y_pred
    residuals_flat = residuals.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    try:
        residual_analysis = analyze_residuals(residuals_flat, fitted_values=y_pred_flat)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_list.append(
            DiagnosticsWarning(
                code="residual_analysis_failed",
                message=str(exc),
            ),
        )
        residual_analysis = None

    contract_uncertainty = uncertainty or UncertaintySummary.point_estimate(provenance=provenance)

    support_level: SupportLevel = "supported"
    if contract_uncertainty.support_level != "supported" or residual_analysis is None:
        support_level = "partial"

    metrics = get_fit_metrics(model, t_arr, y_arr)
    return DiagnosticsContract(
        metrics=metrics,
        residuals=residuals_flat,
        residual_analysis=residual_analysis,
        warnings=warning_list,
        uncertainty=contract_uncertainty,
        support_level=support_level,
        provenance=provenance,
        comparison_family="fitted",
        model_name=model_name,
    )
