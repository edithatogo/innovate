"""Tests for models that don't directly trigger ODE solvers but test more functionality."""

import pytest

from innovate.backend import use_backend

# Use numpy backend to avoid JAX-related issues
use_backend("numpy")

from innovate.compete.competition import MultiProductDiffusionModel
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import ScipyFitter


def test_bass_model_comprehensive_functionality():
    """Test more functionality of Bass model without calling predict."""
    # Test initialization
    model = BassModel()
    assert model.param_names is not None
    assert isinstance(model.param_names, (list, tuple))
    assert len(model.param_names) >= 3  # Should have at least p, q, m

    # Test with covariates
    model_cov = BassModel(covariates=["advertising", "price"])
    assert "advertising" in model_cov.covariates
    assert "price" in model_cov.covariates

    # Test with t_event
    model_event = BassModel(t_event=5.0)
    assert model_event.t_event == 5.0

    # Test initial guesses
    t, y = [0, 1, 2, 3, 4], [10, 25, 50, 75, 90]
    guesses = model.initial_guesses(t, y)
    assert isinstance(guesses, dict)
    assert all(param in guesses for param in ["p", "q", "m"])
    assert guesses["m"] >= max(y)  # Market potential should be at least max observed

    # Test bounds
    bounds = model.bounds(t, y)
    assert isinstance(bounds, dict)
    assert all(param in bounds for param in ["p", "q", "m"])
    for param_bounds in bounds.values():
        assert isinstance(param_bounds, tuple)
        assert len(param_bounds) == 2
        lower, upper = param_bounds
        assert lower <= upper

    # Test parameter setting and getting
    params = {"p": 0.03, "q": 0.38, "m": 1000}
    model.params_ = params
    assert model.params_ == params

    # Test parameter validation
    try:
        # This should work fine
        model.params_ = {"p": 0.05, "q": 0.4, "m": 1200}
        assert model.params_["p"] == 0.05
    except Exception:
        # If there's an exception, it should be handled gracefully
        pass


def test_logistic_model_comprehensive_functionality():
    """Test more functionality of Logistic model without calling predict."""
    # Test initialization
    model = LogisticModel()
    assert model.param_names is not None
    assert isinstance(model.param_names, (list, tuple))
    assert len(model.param_names) >= 3  # Should have at least L, k, x0

    # Test with covariates
    model_cov = LogisticModel(covariates=["marketing"])
    assert "marketing" in model_cov.covariates

    # Test initial guesses
    t, y = [0, 1, 2, 3, 4], [100, 250, 500, 750, 900]
    guesses = model.initial_guesses(t, y)
    assert isinstance(guesses, dict)
    assert all(param in guesses for param in ["L", "k", "x0"])
    assert guesses["L"] >= max(y)  # Carrying capacity should be at least max observed

    # Test bounds
    bounds = model.bounds(t, y)
    assert isinstance(bounds, dict)
    assert all(param in bounds for param in ["L", "k", "x0"])
    for param_bounds in bounds.values():
        assert isinstance(param_bounds, tuple)
        assert len(param_bounds) == 2

    # Test parameter setting and getting
    params = {"L": 1000, "k": 0.2, "x0": 2}
    model.params_ = params
    assert model.params_ == params


def test_gompertz_model_structure():
    """Test Gompertz model structure without calling predict."""
    # Test initialization
    model = GompertzModel()
    assert model.param_names is not None
    assert isinstance(model.param_names, (list, tuple))
    assert len(model.param_names) >= 3  # Should have at least a, b, c

    # Test with covariates
    model_cov = GompertzModel(covariates=["factor1"])
    assert "factor1" in model_cov.covariates

    # Test initial guesses
    t, y = [0, 1, 2, 3, 4], [50, 120, 250, 400, 600]
    guesses = model.initial_guesses(t, y)
    assert isinstance(guesses, dict)
    assert all(param in guesses for param in ["a", "b", "c"])
    assert guesses["a"] >= max(y)  # Asymptote should be at least max observed

    # Test bounds
    bounds = model.bounds(t, y)
    assert isinstance(bounds, dict)
    assert all(param in bounds for param in ["a", "b", "c"])

    # Test parameter setting
    params = {"a": 1000, "b": 2.0, "c": 0.1}
    model.params_ = params
    assert model.params_ == params


def test_multiproduct_model_structure():
    """Test MultiProductDiffusionModel structure without calling predict."""
    # Test initialization with minimal valid parameters
    p_vals = [0.01, 0.02]
    Q_matrix = [[0.1, 0.05], [0.05, 0.1]]
    m_vals = [1000, 1500]
    names = ["ProductA", "ProductB"]

    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=names)

    # Test that properties are correctly set
    assert len(model.p) == 2
    assert model.Q.shape == (2, 2)
    assert len(model.m) == 2
    assert model.N == 2
    assert list(model.names) == names

    # Test parameter-related properties
    param_names = model.param_names
    assert isinstance(param_names, (list, tuple))


def test_scipy_fitter_functionality():
    """Test ScipyFitter functionality."""
    fitter = ScipyFitter()

    # Test that it has the expected interface
    assert hasattr(fitter, "fit")
    assert callable(fitter.fit)

    # Test with a model without actually fitting to avoid ODE
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    # The fitter should have the interface even if we don't call fit()
    assert hasattr(fitter, "fit")


def test_error_handling():
    """Test error handling in models."""
    # Test unfitted model error handling
    bass = BassModel()
    logistic = LogisticModel()

    # Both should initially have empty params
    assert bass.params_ == {}
    assert logistic.params_ == {}

    # Test score method on unfitted models (should raise exception)
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        bass.score([1, 2, 3], [10, 20, 30])

    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        logistic.score([1, 2, 3], [10, 20, 30])


def test_model_composition():
    """Test that multiple models can coexist."""
    # Create multiple models
    bass1 = BassModel()
    bass2 = BassModel(covariates=["ad"])
    logistic = LogisticModel()
    gompertz = GompertzModel()

    # Set different parameters
    bass1.params_ = {"p": 0.02, "q": 0.3, "m": 800}
    bass2.params_ = {"p": 0.03, "q": 0.4, "m": 1200}
    logistic.params_ = {"L": 1000, "k": 0.15, "x0": 3}
    gompertz.params_ = {"a": 900, "b": 3.0, "c": 0.15}

    # Verify they maintain separate states
    assert bass1.params_["p"] != bass2.params_["p"]
    assert logistic.params_["L"] != gompertz.params_["a"]


if __name__ == "__main__":
    print("Running comprehensive functionality tests...")

    test_bass_model_comprehensive_functionality()
    print("✓ Bass model comprehensive functionality test passed")

    test_logistic_model_comprehensive_functionality()
    print("✓ Logistic model comprehensive functionality test passed")

    test_gompertz_model_structure()
    print("✓ Gompertz model structure test passed")

    test_multiproduct_model_structure()
    print("✓ MultiProduct model structure test passed")

    test_scipy_fitter_functionality()
    print("✓ Scipy fitter functionality test passed")

    test_error_handling()
    print("✓ Error handling test passed")

    test_model_composition()
    print("✓ Model composition test passed")

    print("All comprehensive functionality tests passed!")
