"""Sensitivity analysis inputs and deterministic helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SensitivityKind = Literal["parameter", "assumption", "timing", "threshold"]


@dataclass(frozen=True, slots=True)
class ParameterSensitivityInput:
    """Perturb a named scalar parameter around a baseline value."""

    name: str
    baseline: float
    deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.deltas:
            raise ValueError("deltas must be non-empty")
        object.__setattr__(self, "deltas", tuple(float(delta) for delta in self.deltas))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "parameter", "name": self.name, "baseline": self.baseline, "deltas": list(self.deltas)}


@dataclass(frozen=True, slots=True)
class AssumptionSensitivityInput:
    """Toggle or scale a named modeling assumption."""

    name: str
    baseline: float
    alternatives: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.alternatives:
            raise ValueError("alternatives must be non-empty")
        object.__setattr__(self, "alternatives", tuple(float(value) for value in self.alternatives))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "assumption",
            "name": self.name,
            "baseline": float(self.baseline),
            "alternatives": list(self.alternatives),
        }


@dataclass(frozen=True, slots=True)
class TimingSensitivityInput:
    """Shift intervention timing relative to a baseline event time."""

    name: str
    baseline_time: float
    offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.offsets:
            raise ValueError("offsets must be non-empty")
        object.__setattr__(self, "offsets", tuple(float(offset) for offset in self.offsets))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "timing",
            "name": self.name,
            "baseline_time": float(self.baseline_time),
            "offsets": list(self.offsets),
        }


@dataclass(frozen=True, slots=True)
class ThresholdSensitivityInput:
    """Evaluate outcomes against a set of decision thresholds."""

    name: str
    thresholds: tuple[float, ...]
    metric: str = "outcome"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.thresholds:
            raise ValueError("thresholds must be non-empty")
        object.__setattr__(self, "thresholds", tuple(float(value) for value in self.thresholds))
        if not self.metric.strip():
            raise ValueError("metric must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "threshold",
            "name": self.name,
            "thresholds": list(self.thresholds),
            "metric": self.metric,
        }


OutcomeFn = Callable[[Mapping[str, float]], float]


def _json_number(value: float) -> float | None:
    """Return a JSON-safe number; map non-finite values to None (null)."""
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def parameter_perturbation_summary(
    outcome_fn: OutcomeFn,
    inputs: Sequence[ParameterSensitivityInput],
    *,
    context: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic parameter perturbations and elasticities."""
    base_context = dict(context or {})
    rows: list[dict[str, Any]] = []
    for item in inputs:
        baseline = float(item.baseline)
        params = {**base_context, item.name: baseline}
        baseline_outcome = float(outcome_fn(params))
        for delta in item.deltas:
            delta_f = float(delta)
            value = baseline + delta_f
            outcome = float(outcome_fn({**params, item.name: value}))
            abs_change = outcome - baseline_outcome
            if baseline_outcome != 0:
                rel: float | None = abs_change / baseline_outcome
            else:
                rel = None
            if baseline_outcome != 0 and baseline != 0 and delta_f != 0:
                elasticity: float | None = (abs_change / baseline_outcome) / (delta_f / baseline)
            else:
                elasticity = None
            rows.append(
                {
                    "parameter": item.name,
                    "baseline": baseline,
                    "delta": delta_f,
                    "value": value,
                    "baseline_outcome": baseline_outcome,
                    "outcome": outcome,
                    "absolute_change": abs_change,
                    "relative_change": _json_number(rel) if rel is not None else None,
                    "elasticity": _json_number(elasticity) if elasticity is not None else None,
                }
            )
    return {"kind": "parameter_perturbation", "rows": rows, "deterministic": True}


def assumption_sensitivity_summary(
    outcome_fn: OutcomeFn,
    inputs: Sequence[AssumptionSensitivityInput],
    *,
    context: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare outcomes under alternative assumption values."""
    base_context = dict(context or {})
    rows: list[dict[str, Any]] = []
    for item in inputs:
        baseline_outcome = float(outcome_fn({**base_context, item.name: item.baseline}))
        for alternative in item.alternatives:
            outcome = float(outcome_fn({**base_context, item.name: alternative}))
            rows.append(
                {
                    "assumption": item.name,
                    "baseline": item.baseline,
                    "alternative": alternative,
                    "baseline_outcome": baseline_outcome,
                    "outcome": outcome,
                    "absolute_change": outcome - baseline_outcome,
                }
            )
    return {"kind": "assumption_sensitivity", "rows": rows, "deterministic": True}


def intervention_timing_summary(
    outcome_fn: OutcomeFn,
    inputs: Sequence[TimingSensitivityInput],
    *,
    context: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize outcome changes under intervention timing offsets."""
    base_context = dict(context or {})
    rows: list[dict[str, Any]] = []
    for item in inputs:
        baseline_outcome = float(outcome_fn({**base_context, item.name: item.baseline_time}))
        for offset in item.offsets:
            time = item.baseline_time + offset
            outcome = float(outcome_fn({**base_context, item.name: time}))
            rows.append(
                {
                    "intervention": item.name,
                    "baseline_time": item.baseline_time,
                    "offset": offset,
                    "time": time,
                    "baseline_outcome": baseline_outcome,
                    "outcome": outcome,
                    "absolute_change": outcome - baseline_outcome,
                }
            )
    return {"kind": "intervention_timing", "rows": rows, "deterministic": True}


def threshold_sensitivity_summary(
    outcome_values: Sequence[float],
    inputs: Sequence[ThresholdSensitivityInput],
) -> dict[str, Any]:
    """Count how many outcomes meet each threshold (deterministic)."""
    values = np.asarray(list(outcome_values), dtype=float)
    if values.size == 0:
        raise ValueError("outcome_values must be non-empty")
    rows: list[dict[str, Any]] = []
    for item in inputs:
        for threshold in item.thresholds:
            meet = int(np.sum(values >= threshold))
            rows.append(
                {
                    "name": item.name,
                    "metric": item.metric,
                    "threshold": threshold,
                    "n_meet": meet,
                    "n_total": int(values.size),
                    "share_meet": meet / float(values.size),
                }
            )
    return {
        "kind": "threshold_sensitivity",
        "rows": rows,
        "deterministic": True,
        "outcome_summary": {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        },
    }


def combine_sensitivity_summaries(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    """Merge sensitivity blocks for inclusion in a decision report."""
    return {
        "blocks": [dict(summary) for summary in summaries],
        "deterministic": all(bool(summary.get("deterministic", False)) for summary in summaries),
    }
