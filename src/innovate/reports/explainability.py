"""Explainability summaries for adoption, competition, substitution, and policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {str(key): float(value) for key, value in weights.items()}
    if not cleaned:
        raise ValueError("weights must be non-empty")
    total = sum(abs(value) for value in cleaned.values())
    if total == 0:
        n = len(cleaned)
        return dict.fromkeys(cleaned, 1.0 / n)
    return {key: abs(value) / total for key, value in cleaned.items()}


def adoption_driver_summary(
    drivers: Mapping[str, float],
    *,
    baseline_adoption: float,
    scenario_adoption: float,
) -> dict[str, Any]:
    """Attribute adoption change across named drivers (normalized absolute weights)."""
    weights = _normalize_weights(drivers)
    delta = float(scenario_adoption) - float(baseline_adoption)
    contributions = {
        name: {"weight": weight, "contribution": delta * weight} for name, weight in sorted(weights.items())
    }
    return {
        "kind": "adoption_drivers",
        "baseline_adoption": float(baseline_adoption),
        "scenario_adoption": float(scenario_adoption),
        "delta": delta,
        "contributions": contributions,
        "deterministic": True,
    }


def competition_effect_summary(
    product_shares: Mapping[str, float],
    *,
    focal_product: str,
) -> dict[str, Any]:
    """Summarize competitive pressure relative to a focal product share."""
    if focal_product not in product_shares:
        raise ValueError(f"focal_product '{focal_product}' missing from product_shares")
    shares = {str(key): float(value) for key, value in product_shares.items()}
    if any(share < 0.0 for share in shares.values()):
        raise ValueError("product_shares must be non-negative")
    total_share = sum(shares.values())
    if total_share > 1.0 + 1e-9:
        raise ValueError(f"product_shares sum to {total_share}, which exceeds 1.0")
    focal = shares[focal_product]
    competitors = {name: share for name, share in sorted(shares.items()) if name != focal_product}
    pressure = sum(competitors.values())
    return {
        "kind": "competition_effects",
        "focal_product": focal_product,
        "focal_share": focal,
        "competitor_shares": competitors,
        "competitive_pressure": pressure,
        "share_sum": total_share,
        "lead": focal - (max(competitors.values()) if competitors else 0.0),
        "deterministic": True,
    }


def substitution_threshold_summary(
    shares: Sequence[float],
    *,
    thresholds: Sequence[float] = (0.1, 0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Report first time indices where cumulative/share series crosses thresholds."""
    series = np.asarray(list(shares), dtype=float)
    if series.size == 0:
        raise ValueError("shares must be non-empty")
    crossings: list[dict[str, Any]] = []
    for threshold in thresholds:
        idx = np.where(series >= float(threshold))[0]
        crossings.append(
            {
                "threshold": float(threshold),
                "first_index": int(idx[0]) if idx.size else None,
                "met": bool(idx.size),
            }
        )
    return {
        "kind": "substitution_thresholds",
        "crossings": crossings,
        "final_share": float(series[-1]),
        "deterministic": True,
    }


def policy_component_summary(
    components: Mapping[str, float],
    *,
    total_effect: float | None = None,
) -> dict[str, Any]:
    """Decompose a policy effect into named intervention components."""
    parts = {str(key): float(value) for key, value in components.items()}
    if not parts:
        raise ValueError("components must be non-empty")
    summed = sum(parts.values())
    total = float(total_effect) if total_effect is not None else summed
    residual = total - summed
    ordered = sorted(parts.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "kind": "policy_components",
        "components": dict(ordered),
        "sum_components": summed,
        "total_effect": total,
        "residual": residual,
        "dominant_component": ordered[0][0],
        "deterministic": True,
    }


def combine_explainability(*blocks: Mapping[str, Any]) -> dict[str, Any]:
    """Merge explainability blocks for decision-report export."""
    return {
        "blocks": [dict(block) for block in blocks],
        "deterministic": all(bool(block.get("deterministic", False)) for block in blocks),
    }
