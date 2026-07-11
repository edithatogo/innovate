"""Tests for the plugin subsystem."""

from typing import Generator

import pytest

from innovate.plugins import (
    _REGISTERED_EXTENSIONS,
    ExtensionManifest,
    get_registered_extensions,
    register_extension,
)


@pytest.fixture(autouse=True)
def isolate_registry() -> Generator[None]:
    """Isolate the global plugin registry for each test."""
    original = _REGISTERED_EXTENSIONS.copy()
    _REGISTERED_EXTENSIONS.clear()
    try:
        yield
    finally:
        _REGISTERED_EXTENSIONS.clear()
        _REGISTERED_EXTENSIONS.update(original)


def test_register_extension_success() -> None:
    """It should register a valid extension manifest."""
    manifest = ExtensionManifest(
        name="test-plugin",
        version="1.0.0",
        entrypoint="test.plugin:register",
        stability="stable",
        extension_points=("model_registry",),
    )
    register_extension(manifest)

    registry = get_registered_extensions()
    assert "test-plugin" in registry
    assert registry["test-plugin"] is manifest


def test_register_extension_duplicate_raises() -> None:
    """It should raise ValueError when registering a duplicate extension name."""
    manifest = ExtensionManifest(
        name="test-plugin",
        version="1.0.0",
        entrypoint="test.plugin:register",
        stability="stable",
        extension_points=("model_registry",),
    )
    register_extension(manifest)

    with pytest.raises(ValueError, match="is already registered"):
        register_extension(manifest)


def test_register_extension_validates() -> None:
    """It should validate the manifest before registering it."""
    manifest = ExtensionManifest(
        name="test-plugin",
        version="1.0.0",
        entrypoint="test.plugin:register",
        stability="stable",
        extension_points=("model_registry",),
    )

    # Mutate to an invalid state to prove validate() is called
    object.__setattr__(manifest, "name", "")

    with pytest.raises(ValueError, match="name must be non-empty"):
        register_extension(manifest)

    registry = get_registered_extensions()
    assert "test-plugin" not in registry


def test_manifest_validation_empty_version() -> None:
    """It should raise ValueError for an empty version."""
    with pytest.raises(ValueError, match="version must be non-empty"):
        ExtensionManifest(
            name="test-plugin",
            version="",
            entrypoint="test.plugin:register",
            stability="stable",
            extension_points=("model_registry",),
        )


def test_manifest_validation_invalid_entrypoint() -> None:
    """It should raise ValueError for an invalid entrypoint."""
    with pytest.raises(ValueError, match="entrypoint must be in 'module:callable' format"):
        ExtensionManifest(
            name="test-plugin",
            version="1.0.0",
            entrypoint="test.plugin.register",
            stability="stable",
            extension_points=("model_registry",),
        )
    with pytest.raises(ValueError, match="entrypoint must be in 'module:callable' format"):
        ExtensionManifest(
            name="test-plugin",
            version="1.0.0",
            entrypoint="",
            stability="stable",
            extension_points=("model_registry",),
        )


def test_manifest_validation_empty_extension_points() -> None:
    """It should raise ValueError for empty extension points."""
    with pytest.raises(ValueError, match="extension_points must be non-empty"):
        ExtensionManifest(
            name="test-plugin",
            version="1.0.0",
            entrypoint="test.plugin:register",
            stability="stable",
            extension_points=(),
        )


def test_manifest_validation_unknown_extension_points() -> None:
    """It should raise ValueError for unknown extension points."""
    with pytest.raises(ValueError, match="Unknown extension points: unknown_point"):
        ExtensionManifest(
            name="test-plugin",
            version="1.0.0",
            entrypoint="test.plugin:register",
            stability="stable",
            extension_points=("model_registry", "unknown_point"),
        )
