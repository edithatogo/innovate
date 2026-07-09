"""Shared diagnostics and uncertainty contract for model fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters.residual_analysis import ResidualAnalysis, analyze_residuals

WarningSeverity = Literal["info", "warning", "error"]
ReportType = Literal["point_estimate", "bootstrap_interval", "posterior_summary", "unsupported"]
SupportLevel = Literal["supported", "partial", "unsupported"]
Provenance = Literal["deterministic", "bootstrap", "bayesian", "unknown"]
DiagnosticsArtifactKind = Literal[
    "residual_diagnostics",
    "calibration_check",
    "uncertainty_interval",
    "model_comparison",
]

DIAGNOSTICS_ARTIFACT_SCHEMA_MAJOR_VERSION = 1
DIAGNOSTICS_ARTIFACT_SCHEMA_MINOR_VERSION = 0
DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION = (
    f"{DIAGNOSTICS_ARTIFACT_SCHEMA_MAJOR_VERSION}.{DIAGNOSTICS_ARTIFACT_SCHEMA_MINOR_VERSION}"
)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


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


@dataclass(frozen=True)
class DiagnosticsArtifactPayload:
    """Versioned, schema-compatible diagnostics artifacts for language bindings."""

    model_name: str
    support_level: SupportLevel
    provenance: Provenance
    artifacts: dict[str, dict[str, Any]]
    schema_version: str = DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION
    backend: str = "numpy"
    xla_eligible: bool = False
    xla_rationale: str = (
        "The artifact contract is currently assembled from deterministic NumPy/Python diagnostics. "
        "Array-heavy residual and interval kernels can be promoted to JAX/XLA after parity and "
        "benchmark gates pass."
    )
    promotion_criteria: tuple[str, ...] = (
        "schema-compatible payload",
        "deterministic or tolerance-bounded tests",
        "binding fixture coverage",
        "documented support tier",
    )

    def __post_init__(self) -> None:
        if self.schema_version != DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported diagnostics artifact schema version: {self.schema_version}",
            )
        if self.support_level not in {"supported", "partial", "unsupported"}:
            raise ValueError("Diagnostics artifact support_level is invalid")
        if self.provenance not in {"deterministic", "bootstrap", "bayesian", "unknown"}:
            raise ValueError("Diagnostics artifact provenance is invalid")
        if not self.artifacts:
            raise ValueError("Diagnostics artifact payloads must include at least one artifact")

    @classmethod
    def from_contract(cls, contract: DiagnosticsContract) -> DiagnosticsArtifactPayload:
        """Build the stable artifact payload from the existing diagnostics contract."""
        residual_rows = [
            {
                "index": int(index),
                "residual": float(residual),
                "standardized_residual": float(standardized),
            }
            for index, (residual, standardized) in enumerate(
                zip(
                    contract.residuals.tolist(),
                    (
                        contract.residual_analysis.standardized_residuals.tolist()
                        if contract.residual_analysis is not None
                        else np.zeros_like(contract.residuals, dtype=float).tolist()
                    ),
                    strict=True,
                ),
            )
        ]

        residual_summary: dict[str, float | None] = {}
        if contract.residual_analysis is not None:
            residual_summary = {
                "mean_residual": _finite_or_none(contract.residual_analysis.mean_residual),
                "std_residual": _finite_or_none(contract.residual_analysis.std_residual),
                "max_abs_residual": _finite_or_none(contract.residual_analysis.max_abs_residual),
                "durbin_watson": _finite_or_none(contract.residual_analysis.durbin_watson),
                "shapiro_wilk_p": _finite_or_none(contract.residual_analysis.shapiro_wilk_p),
                "breusch_pagan_p": _finite_or_none(contract.residual_analysis.breusch_pagan_p),
            }

        uncertainty_rows = [
            {
                "parameter": name,
                "lower": _finite_or_none(contract.uncertainty.lower.get(name)),
                "median": _finite_or_none(contract.uncertainty.median.get(name)),
                "upper": _finite_or_none(contract.uncertainty.upper.get(name)),
            }
            for name in sorted(
                set(contract.uncertainty.lower) | set(contract.uncertainty.median) | set(contract.uncertainty.upper),
            )
        ]

        metric_rows = [
            {"metric": name, "value": _finite_or_none(value)} for name, value in sorted(contract.metrics.items())
        ]

        return cls(
            model_name=contract.model_name,
            support_level=contract.support_level,
            provenance=contract.provenance,
            artifacts={
                "residuals": {
                    "kind": "residual_diagnostics",
                    "support_level": contract.support_level,
                    "columns": ("index", "residual", "standardized_residual"),
                    "rows": residual_rows,
                    "summary": residual_summary,
                    "arrow_compatible": True,
                },
                "calibration": {
                    "kind": "calibration_check",
                    "support_level": "partial" if residual_summary else "unsupported",
                    "summary": {
                        "mean_residual": residual_summary.get("mean_residual"),
                        "max_abs_residual": residual_summary.get("max_abs_residual"),
                    },
                    "note": "Initial calibration slice uses residual bias and magnitude checks.",
                    "arrow_compatible": True,
                },
                "uncertainty": {
                    "kind": "uncertainty_interval",
                    "support_level": contract.uncertainty.support_level,
                    "report_type": contract.uncertainty.report_type,
                    "level": contract.uncertainty.level,
                    "columns": ("parameter", "lower", "median", "upper"),
                    "rows": uncertainty_rows,
                    "arrow_compatible": True,
                },
                "model_comparison": {
                    "kind": "model_comparison",
                    "support_level": contract.support_level,
                    "comparison_family": contract.comparison_family,
                    "columns": ("metric", "value"),
                    "rows": metric_rows,
                    "arrow_compatible": True,
                },
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics artifacts to JSON-friendly values."""
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "support_level": self.support_level,
            "provenance": self.provenance,
            "backend": self.backend,
            "xla": {
                "eligible": self.xla_eligible,
                "rationale": self.xla_rationale,
            },
            "promotion_criteria": list(self.promotion_criteria),
            "artifacts": self.artifacts,
        }

    def to_table_payloads(self) -> dict[str, Any]:
        """Return Arrow-friendly kernel table payloads for tabular artifacts."""
        import operator

        from innovate.kernel import KernelTablePayload  # Local import avoids an import cycle.

        tables: dict[str, KernelTablePayload] = {}
        for name, artifact in self.artifacts.items():
            columns = artifact.get("columns")
            rows = artifact.get("rows")
            if not columns or not isinstance(rows, list):
                continue

            none_tuple = tuple(None for _ in columns)
            try:
                getter = operator.itemgetter(*columns)
                if len(columns) == 1:
                    table_rows = [(getter(row),) if isinstance(row, dict) else none_tuple for row in rows]
                else:
                    table_rows = [getter(row) if isinstance(row, dict) else none_tuple for row in rows]
            except KeyError:
                table_rows = [
                    tuple(row.get(column) if isinstance(row, dict) else None for column in columns) for row in rows
                ]
            tables[name] = KernelTablePayload.from_rows(
                columns=tuple(str(column) for column in columns),
                rows=table_rows,
                metadata={
                    "diagnostics_artifact": name,
                    "diagnostics_artifact_kind": str(artifact.get("kind", "")),
                    "diagnostics_artifact_schema_version": self.schema_version,
                    "model_name": self.model_name,
                    "support_level": self.support_level,
                },
            )
        return tables


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

    def to_artifact_payload(self) -> DiagnosticsArtifactPayload:
        """Build the versioned diagnostics artifact payload."""
        return DiagnosticsArtifactPayload.from_contract(self)

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
            "artifacts": self.to_artifact_payload().to_dict(),
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
    t_values = t_arr.tolist()
    y_values = y_arr.tolist()
    y_pred = np.asarray(model.predict(t_values), dtype=float)
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

    metrics = get_fit_metrics(model, t_values, y_values)
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
