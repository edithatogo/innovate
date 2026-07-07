"""Tests for stability lifecycle guidance."""

from __future__ import annotations

import pytest

from innovate.stability import (
    STABILITY_LIFECYCLE_RULES,
    StabilityTier,
    describe_stability_tier,
)


def test_describe_stability_tier_with_enum() -> None:
    """Test describe_stability_tier with StabilityTier enum values."""
    assert describe_stability_tier(StabilityTier.STABLE) == STABILITY_LIFECYCLE_RULES[StabilityTier.STABLE]
    assert describe_stability_tier(StabilityTier.PROVISIONAL) == STABILITY_LIFECYCLE_RULES[StabilityTier.PROVISIONAL]
    assert describe_stability_tier(StabilityTier.INTERNAL) == STABILITY_LIFECYCLE_RULES[StabilityTier.INTERNAL]


def test_describe_stability_tier_with_strings() -> None:
    """Test describe_stability_tier with string aliases."""
    assert describe_stability_tier("stable") == STABILITY_LIFECYCLE_RULES[StabilityTier.STABLE]
    assert describe_stability_tier("beta") == STABILITY_LIFECYCLE_RULES[StabilityTier.PROVISIONAL]
    assert describe_stability_tier("experimental") == STABILITY_LIFECYCLE_RULES[StabilityTier.PROVISIONAL]
    assert describe_stability_tier("provisional") == STABILITY_LIFECYCLE_RULES[StabilityTier.PROVISIONAL]
    assert describe_stability_tier("internal") == STABILITY_LIFECYCLE_RULES[StabilityTier.INTERNAL]
    assert describe_stability_tier("internal-only") == STABILITY_LIFECYCLE_RULES[StabilityTier.INTERNAL]

    # Test whitespace and case insensitivity
    assert describe_stability_tier("  STABLE  ") == STABILITY_LIFECYCLE_RULES[StabilityTier.STABLE]
    assert describe_stability_tier("BETA") == STABILITY_LIFECYCLE_RULES[StabilityTier.PROVISIONAL]


def test_describe_stability_tier_invalid_value() -> None:
    """Test describe_stability_tier raises ValueError for unknown tiers."""
    with pytest.raises(ValueError, match="Unknown stability tier 'unknown'"):
        describe_stability_tier("unknown")

    with pytest.raises(ValueError, match="Unknown stability tier 'production'"):
        describe_stability_tier("production")
