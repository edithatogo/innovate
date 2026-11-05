"""Tests for stability lifecycle guidance."""

from __future__ import annotations

import pytest

from innovate.stability import (
    STABILITY_LIFECYCLE_RULES,
    StabilityTier,
    describe_stability_tier,
    normalize_stability_tier,
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


def test_stability_tier_enum_values() -> None:
    """Test that StabilityTier enum has expected values and behaves like strings."""
    assert StabilityTier.STABLE == "stable"
    assert StabilityTier.PROVISIONAL == "provisional"
    assert StabilityTier.INTERNAL == "internal"

    assert str(StabilityTier.STABLE) == "stable"

    # Ensure all enum values are mapped in the rules
    for tier in StabilityTier:
        assert tier in STABILITY_LIFECYCLE_RULES
        assert isinstance(STABILITY_LIFECYCLE_RULES[tier], str)


def test_stability_aliases_consistency() -> None:
    """Test that all aliases map to valid StabilityTier members."""
    from innovate.stability import _STABILITY_ALIASES

    for alias, tier in _STABILITY_ALIASES.items():
        assert isinstance(alias, str)
        assert isinstance(tier, StabilityTier)
        assert alias.islower()


def test_stability_tier_instantiation() -> None:
    """Test instantiating StabilityTier directly from strings."""
    assert StabilityTier("stable") is StabilityTier.STABLE
    assert StabilityTier("provisional") is StabilityTier.PROVISIONAL
    assert StabilityTier("internal") is StabilityTier.INTERNAL

    with pytest.raises(ValueError, match="is not a valid StabilityTier"):
        StabilityTier("unknown")


def test_stability_lifecycle_rules_completeness() -> None:
    """Test that all StabilityTier members have lifecycle rules and vice versa."""
    # Ensure every tier has a lifecycle rule
    for tier in StabilityTier:
        assert tier in STABILITY_LIFECYCLE_RULES

    # Ensure there are no extra rules for undefined tiers
    for tier in STABILITY_LIFECYCLE_RULES:
        assert isinstance(tier, StabilityTier)
        assert tier in StabilityTier


def test_stability_aliases_exhaustive() -> None:
    """Test that every StabilityTier can be reached via an alias."""
    from innovate.stability import _STABILITY_ALIASES

    # Collect all unique tiers targeted by aliases
    targeted_tiers = set(_STABILITY_ALIASES.values())

    # Ensure all actual enum members are targeted
    for tier in StabilityTier:
        assert tier in targeted_tiers
