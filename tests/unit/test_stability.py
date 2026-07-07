"""Tests for stability tier normalization and lifecycle guidance."""

from __future__ import annotations

import pytest

from innovate.stability import StabilityTier, describe_stability_tier, normalize_stability_tier


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (StabilityTier.STABLE, StabilityTier.STABLE),
        (StabilityTier.PROVISIONAL, StabilityTier.PROVISIONAL),
        (StabilityTier.INTERNAL, StabilityTier.INTERNAL),
        ("stable", StabilityTier.STABLE),
        (" STABLE ", StabilityTier.STABLE),
        ("beta", StabilityTier.PROVISIONAL),
        ("BETA", StabilityTier.PROVISIONAL),
        ("experimental", StabilityTier.PROVISIONAL),
        ("provisional", StabilityTier.PROVISIONAL),
        ("internal", StabilityTier.INTERNAL),
        ("internal-only", StabilityTier.INTERNAL),
        (" InTeRnAl-OnLy ", StabilityTier.INTERNAL),
    ],
)
def test_normalize_stability_tier_success(value: str | StabilityTier, expected: StabilityTier) -> None:
    """It should correctly normalize valid strings and enums to StabilityTier."""
    assert normalize_stability_tier(value) is expected


def test_normalize_stability_tier_error() -> None:
    """It should raise ValueError for unknown stability tiers."""
    with pytest.raises(ValueError, match="Unknown stability tier 'unknown'"):
        normalize_stability_tier("unknown")


def test_describe_stability_tier() -> None:
    """It should return the lifecycle guidance for a valid tier."""
    description = describe_stability_tier("beta")
    assert "Provisional surfaces are intended for active iteration" in description
