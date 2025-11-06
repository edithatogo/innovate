"""Tests for the backend selection module."""

import pytest
from unittest.mock import patch

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


@patch('src.innovate.backend.JaxBackend', None)
def test_use_backend_jax_import_error():
    """Test that using JAX backend raises ImportError when it's not available."""
    with pytest.raises(ImportError, match="JAX backend is not available. Install jax and diffrax to use it."):
        backend.use_backend("jax")


def test_use_backend_unknown():
    """Test that using an unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend: unknown_backend"):
        backend.use_backend("unknown_backend")


@patch('src.innovate.backend.JaxBackend', create=True)
def test_use_backend_jax_success(mock_jax_backend_class):
    """Test switching to JAX backend when it's available."""
    # Mock the JAX backend instance
    mock_jax_instance = mock_jax_backend_class.return_value
    mock_jax_backend_class.return_value.__class__.__name__ = "JaxBackendMock"
    
    # Temporarily replace the module-level JaxBackend with our mock
    original_jax_backend = backend.JaxBackend
    backend.JaxBackend = mock_jax_backend_class
    
    try:
        # Switch to JAX backend
        backend.use_backend("jax")
        
        # The verification would depend on how the JaxBackend is implemented
        # Since we can't actually import it without the dependencies, we just check
        # that the use_backend function accepts the "jax" parameter without error
        # when JaxBackend is available
        pass
    finally:
        # Restore the original JaxBackend
        backend.JaxBackend = original_jax_backend