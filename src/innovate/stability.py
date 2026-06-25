"""Stability tiers and lifecycle guidance for :mod:`innovate`."""

from __future__ import annotations

from enum import StrEnum


class StabilityTier(StrEnum):
    """Normalized stability tiers for stable, provisional, and internal surfaces."""

    STABLE = "stable"
    PROVISIONAL = "provisional"
    INTERNAL = "internal"


_STABILITY_ALIASES: dict[str, StabilityTier] = {
    "stable": StabilityTier.STABLE,
    "beta": StabilityTier.PROVISIONAL,
    "experimental": StabilityTier.PROVISIONAL,
    "provisional": StabilityTier.PROVISIONAL,
    "internal": StabilityTier.INTERNAL,
    "internal-only": StabilityTier.INTERNAL,
}


STABILITY_LIFECYCLE_RULES: dict[StabilityTier, str] = {
    StabilityTier.STABLE: (
        "Stable surfaces are versioned public contract points. "
        "They MAY receive deprecations only with a documented migration window."
    ),
    StabilityTier.PROVISIONAL: (
        "Provisional surfaces are intended for active iteration. "
        "They MAY promote to stable after compatibility and coverage are proven, "
        "and MAY be removed or reshaped with advance notice."
    ),
    StabilityTier.INTERNAL: (
        "Internal surfaces are implementation details. "
        "They MUST NOT be treated as public contract and MAY change without notice."
    ),
}


def normalize_stability_tier(value: str | StabilityTier) -> StabilityTier:
    """Map a string or tier enum to the normalized stability vocabulary."""
    if isinstance(value, StabilityTier):
        return value

    try:
        return _STABILITY_ALIASES[value.strip().lower()]
    except KeyError as exc:
        available = ", ".join(sorted({tier.value for tier in StabilityTier}))
        raise ValueError(
            f"Unknown stability tier {value!r}. Available tiers: {available}.",
        ) from exc


def describe_stability_tier(value: str | StabilityTier) -> str:
    """Return the lifecycle guidance associated with a tier."""
    tier = normalize_stability_tier(value)
    return STABILITY_LIFECYCLE_RULES[tier]
