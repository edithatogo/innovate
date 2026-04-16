"""Tests for the backend selection module."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.innovate import backend


def test_backend_initialization():
    """Test that the backend is initialized with NumPy by default."""
    # Check that the default backend is NumPy
    assert backend.current_backend.__class__.__name__ == "NumPyBackend"


def test_use_backend_numpy():
    """Test switching to NumPy backend."""
    # Switch to NumPy backend
    backend.use_backend("numpy")

    # Verify the backend is set to NumPy
    assert backend.current_backend.__class__.__name__ == "NumPyBackend"


@patch("src.innovate.backend.JaxBackend", None)
def test_use_backend_jax_import_error():
    """Test that using JAX backend raises ImportError when it's not available."""
    with pytest.raises(ImportError, match="JAX backend is not available. Install innovate\\[jax\\] to enable it."):
        backend.use_backend("jax")


def test_use_backend_unknown():
    """Test that using an unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend: unknown_backend"):
        backend.use_backend("unknown_backend")


class FakeJaxBackend:
    """Test double for the optional JAX backend."""


@patch("src.innovate.backend.JaxBackend", FakeJaxBackend)
def test_use_backend_jax_success():
    """Test switching to JAX backend when it's available."""
    with patch.object(backend, "get_backend_capability", return_value=SimpleNamespace(available=True)):
        backend.use_backend("jax")
        assert backend.current_backend.__class__.__name__ == "FakeJaxBackend"
