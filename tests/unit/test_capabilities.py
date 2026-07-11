import pytest

from innovate.capabilities import (
    get_backend_capability,
    get_backend_registry,
    get_fitter_capability,
    get_fitter_registry,
    get_model_capability,
)


def test_get_model_capability_error_path():
    """Test that get_model_capability raises a KeyError with available models."""
    with pytest.raises(KeyError, match=r"Unknown model capability 'nonexistent'\. Available models:.*bass.*"):
        get_model_capability("nonexistent")


def test_get_backend_capability_error_path():
    """Test that get_backend_capability raises a KeyError with available backends."""
    with pytest.raises(KeyError, match=r"Unknown backend capability 'nonexistent'\. Available backends:.*jax.*numpy.*"):
        get_backend_capability("nonexistent")


def test_get_fitter_capability_error_path():
    """Test that get_fitter_capability raises a KeyError with available fitters."""
    with pytest.raises(KeyError, match=r"Unknown fitter capability 'nonexistent'\. Available fitters:.*scipy.*"):
        get_fitter_capability("nonexistent")


def test_get_backend_registry():
    """Test getting the backend registry."""
    registry = get_backend_registry()
    assert "numpy" in registry
    assert "jax" in registry


def test_get_fitter_registry():
    """Test getting the fitter registry."""
    registry = get_fitter_registry()
    assert "scipy" in registry
    assert "bootstrap" in registry


def test_stability_tier():
    """Test stability_tier on capability models."""
    model = get_model_capability("bass")
    assert model.stability_tier == "stable"

    backend = get_backend_capability("numpy")
    assert backend.stability_tier == "stable"

    fitter = get_fitter_capability("scipy")
    assert fitter.stability_tier == "stable"


def test_get_model_registry():
    """Test getting the model registry."""
    from innovate.capabilities import get_model_registry

    registry = get_model_registry()
    assert "bass" in registry
    assert "logistic" in registry
