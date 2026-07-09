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
