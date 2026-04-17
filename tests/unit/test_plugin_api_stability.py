"""Tests for plugin API contracts and stability tiers."""

from __future__ import annotations

import pytest

import innovate
from innovate.capabilities import ModelCapability
from innovate.plugins import (
    ExtensionManifest,
    StabilityTier,
    get_registered_extensions,
    normalize_stability_tier,
    register_extension,
)


def test_capabilities_expose_normalized_stability_tiers() -> None:
    """Capability metadata should normalize into a small tier vocabulary."""
    registry = innovate.get_model_registry()

    assert innovate.StabilityTier is StabilityTier
    assert innovate.normalize_stability_tier is normalize_stability_tier
    assert innovate.ExtensionManifest is ExtensionManifest

    assert normalize_stability_tier("stable") is StabilityTier.STABLE
    assert normalize_stability_tier("experimental") is StabilityTier.PROVISIONAL
    assert normalize_stability_tier("internal") is StabilityTier.INTERNAL

    assert isinstance(registry["bass"], ModelCapability)
    assert registry["bass"].stability_tier is StabilityTier.STABLE
    assert registry["hierarchical"].stability_tier is StabilityTier.PROVISIONAL

    with pytest.raises(ValueError, match="Unknown stability tier"):
        normalize_stability_tier("beta-release")


def test_extension_manifest_registers_and_validates() -> None:
    """A manifest should validate before it can be registered."""
    manifest = ExtensionManifest(
        name="demo-plugin",
        version="1.0.0",
        entrypoint="demo.plugin:register",
        stability=StabilityTier.PROVISIONAL,
        extension_points=("model_registry", "diagnostics"),
    )

    assert manifest.validate() is None
    register_extension(manifest)

    registry = get_registered_extensions()
    assert registry["demo-plugin"] is manifest
    assert registry["demo-plugin"].stability is StabilityTier.PROVISIONAL
    assert registry["demo-plugin"].extension_points == ("model_registry", "diagnostics")

    with pytest.raises(ValueError, match="already registered"):
        register_extension(manifest)


def test_extension_manifest_rejects_unknown_extension_points() -> None:
    """Manifests should fail fast on unsupported extension points."""
    with pytest.raises(ValueError, match="Unknown extension points"):
        ExtensionManifest(
            name="broken-plugin",
            version="1.0.0",
            entrypoint="broken.plugin:register",
            stability=StabilityTier.INTERNAL,
            extension_points=("model_registry", "remote_distribution"),
        )
