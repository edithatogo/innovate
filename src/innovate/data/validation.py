"""Validation diagnostics for dataset contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from innovate.data.contracts import (
    AdoptionDataset,
    CompetitionDataset,
    DatasetContract,
    NetworkEdgeDataset,
    PolicyTimingDataset,
    SubstitutionDataset,
)

CheckStatus = Literal["pass", "warn", "fail"]
SEVERITY_ORDER = {"pass": 0, "warn": 1, "fail": 2}


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Single validation diagnostic result."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate validation report for a dataset contract."""

    kind: str
    checks: tuple[ValidationCheck, ...]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "failed": [check.name for check in self.checks if check.status == "fail"],
            "warnings": [check.name for check in self.checks if check.status == "warn"],
        }


def _missingness(values: np.ndarray, name: str) -> ValidationCheck:
    # Finite arrays are required by contracts; this guard still reports density.
    missing = int(np.isnan(values).sum()) if values.dtype.kind == "f" else 0
    total = int(values.size)
    ratio = missing / total if total else 1.0
    if missing:
        return ValidationCheck(
            name=f"missingness:{name}",
            status="fail",
            message=f"{missing}/{total} missing values in {name}",
            details={"missing": missing, "total": total, "ratio": ratio},
        )
    return ValidationCheck(
        name=f"missingness:{name}",
        status="pass",
        message=f"no missing values in {name}",
        details={"missing": 0, "total": total, "ratio": 0.0},
    )


def _time_alignment(time: np.ndarray) -> ValidationCheck:
    if time.size < 2:
        return ValidationCheck(
            name="time_alignment",
            status="pass",
            message="single observation; time alignment not applicable",
            details={"count": int(time.size)},
        )
    diffs = np.diff(time)
    if np.any(diffs <= 0):
        return ValidationCheck(
            name="time_alignment",
            status="fail",
            message="time values must be strictly increasing",
            details={"non_positive_steps": int(np.sum(diffs <= 0))},
        )
    return ValidationCheck(
        name="time_alignment",
        status="pass",
        message="time values are strictly increasing",
        details={"min_step": float(diffs.min()), "max_step": float(diffs.max())},
    )


def _monotonicity(values: np.ndarray, name: str, *, non_decreasing: bool = True) -> ValidationCheck:
    if values.size < 2:
        return ValidationCheck(
            name=f"monotonicity:{name}",
            status="pass",
            message=f"{name} monotonicity not applicable for single value",
        )
    diffs = np.diff(values)
    ok = bool(np.all(diffs >= -1e-12)) if non_decreasing else bool(np.all(diffs <= 1e-12))
    if not ok:
        return ValidationCheck(
            name=f"monotonicity:{name}",
            status="fail",
            message=f"{name} is not {'non-decreasing' if non_decreasing else 'non-increasing'}",
            details={"violations": int(np.sum(diffs < -1e-12 if non_decreasing else diffs > 1e-12))},
        )
    return ValidationCheck(
        name=f"monotonicity:{name}",
        status="pass",
        message=f"{name} is {'non-decreasing' if non_decreasing else 'non-increasing'}",
    )


def _duplicate_keys(keys: Sequence[tuple[Any, ...]], name: str) -> ValidationCheck:
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    if duplicates:
        return ValidationCheck(
            name=f"duplicates:{name}",
            status="fail",
            message=f"{duplicates} duplicate keys in {name}",
            details={"duplicates": duplicates, "unique": len(seen)},
        )
    return ValidationCheck(
        name=f"duplicates:{name}",
        status="pass",
        message=f"no duplicate keys in {name}",
        details={"duplicates": 0, "unique": len(seen)},
    )


def _denominator_consistency(adoption: np.ndarray, denominator: np.ndarray | None) -> ValidationCheck:
    if denominator is None:
        return ValidationCheck(
            name="denominator_consistency",
            status="pass",
            message="no denominator provided",
        )
    if np.any(adoption > denominator + 1e-9):
        return ValidationCheck(
            name="denominator_consistency",
            status="fail",
            message="adoption exceeds denominator for one or more observations",
            details={"violations": int(np.sum(adoption > denominator + 1e-9))},
        )
    return ValidationCheck(
        name="denominator_consistency",
        status="pass",
        message="adoption is within denominator bounds",
    )


def _unit_compatibility(unit: str, allowed: set[str], name: str) -> ValidationCheck:
    if unit not in allowed:
        return ValidationCheck(
            name=f"unit_compatibility:{name}",
            status="fail",
            message=f"unit '{unit}' is not compatible for {name}; allowed={sorted(allowed)}",
            details={"unit": unit, "allowed": sorted(allowed)},
        )
    return ValidationCheck(
        name=f"unit_compatibility:{name}",
        status="pass",
        message=f"unit '{unit}' is compatible for {name}",
        details={"unit": unit},
    )


def _adoption_share_bounds(dataset: AdoptionDataset) -> ValidationCheck | None:
    if dataset.unit != "share":
        return None
    if np.any((dataset.adoption < 0.0) | (dataset.adoption > 1.0 + 1e-9)):
        return ValidationCheck(
            name="share_bounds:adoption",
            status="fail",
            message="adoption values with unit='share' must be in [0, 1]",
            details={
                "min": float(dataset.adoption.min()),
                "max": float(dataset.adoption.max()),
            },
        )
    return ValidationCheck(
        name="share_bounds:adoption",
        status="pass",
        message="adoption share values are within [0, 1]",
    )


def _validate_adoption(dataset: AdoptionDataset) -> list[ValidationCheck]:
    checks = [
        _missingness(dataset.time, "time"),
        _missingness(dataset.adoption, "adoption"),
        _time_alignment(dataset.time),
        _monotonicity(dataset.adoption, "adoption", non_decreasing=True),
        _denominator_consistency(dataset.adoption, dataset.denominator),
        _duplicate_keys([(float(t),) for t in dataset.time], "time"),
        _unit_compatibility(dataset.unit, {"count", "share", "rate"}, "adoption"),
    ]
    if dataset.denominator is not None:
        checks.append(_missingness(dataset.denominator, "denominator"))
    share_check = _adoption_share_bounds(dataset)
    if share_check is not None:
        checks.append(share_check)
    return checks


def validate_dataset(dataset: DatasetContract) -> ValidationReport:
    """Run standard validation diagnostics for a dataset contract."""
    checks: list[ValidationCheck] = []
    if isinstance(dataset, AdoptionDataset):
        checks.extend(_validate_adoption(dataset))
    elif isinstance(dataset, SubstitutionDataset):
        checks.extend(
            [
                _missingness(dataset.time, "time"),
                _missingness(dataset.share.reshape(-1), "share"),
                _time_alignment(dataset.time),
                _duplicate_keys([(float(t),) for t in dataset.time], "time"),
                _unit_compatibility(dataset.unit, {"share"}, "substitution"),
            ]
        )
        row_sums = dataset.share.sum(axis=1)
        if np.any(row_sums > 1.0 + 1e-9):
            checks.append(
                ValidationCheck(
                    name="share_sum",
                    status="fail",
                    message="row share sums exceed 1.0",
                    details={"max_sum": float(row_sums.max())},
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="share_sum",
                    status="pass",
                    message="row share sums are within [0, 1]",
                    details={"max_sum": float(row_sums.max())},
                )
            )
    elif isinstance(dataset, CompetitionDataset):
        checks.extend(
            [
                _missingness(dataset.time, "time"),
                _missingness(dataset.value, "value"),
                _duplicate_keys(
                    list(zip(dataset.time.tolist(), dataset.unit_id, dataset.product_id, strict=True)),
                    "time_unit_product",
                ),
                _unit_compatibility(dataset.unit, {"count", "share", "rate", "currency"}, "competition"),
            ]
        )
    elif isinstance(dataset, PolicyTimingDataset):
        checks.extend(
            [
                _missingness(dataset.event_times, "event_times"),
                _missingness(dataset.event_effects, "event_effects"),
                _duplicate_keys(
                    list(zip(dataset.event_times.tolist(), dataset.event_labels, strict=True)),
                    "event_time_label",
                ),
                _unit_compatibility(dataset.unit, {"effect", "rate", "count"}, "policy_timing"),
            ]
        )
        if dataset.event_times.size >= 2 and np.any(np.diff(dataset.event_times) < 0):
            checks.append(
                ValidationCheck(
                    name="event_time_order",
                    status="warn",
                    message="event_times are not sorted; consumers may reorder",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="event_time_order",
                    status="pass",
                    message="event_times are sorted or single-valued",
                )
            )
    elif isinstance(dataset, NetworkEdgeDataset):
        checks.extend(
            [
                _missingness(dataset.weight, "weight"),
                _duplicate_keys(
                    list(zip(dataset.source, dataset.target, strict=True)),
                    "source_target",
                ),
                _unit_compatibility(dataset.unit, {"weight", "count", "rate"}, "network_edges"),
            ]
        )
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported dataset type: {type(dataset)!r}")

    ok = all(check.status != "fail" for check in checks)
    return ValidationReport(kind=dataset.kind, checks=tuple(checks), ok=ok)


def require_valid(dataset: DatasetContract) -> ValidationReport:
    """Validate and raise ValueError when any check fails."""
    report = validate_dataset(dataset)
    if not report.ok:
        failed = ", ".join(check.name for check in report.checks if check.status == "fail")
        raise ValueError(f"dataset validation failed: {failed}")
    return report
