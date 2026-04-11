"""Tests for the multi-product competition module."""

import numpy as np
import pytest

from src.innovate.compete.multi_product import MultiProductDiffusionModel


def test_multi_product_diffusion_model_init():
    """Test initializing the MultiProductDiffusionModel."""
    # Test with valid number of products
    model = MultiProductDiffusionModel(n_products=2)

    assert model.n_products == 2
    assert model._params == {}
    assert model.covariates == []
    assert model.p is None
    assert model.Q is None
    assert model.m is None
    assert model.names is None


def test_multi_product_diffusion_model_init_invalid_n_products():
    """Test initializing with invalid number of products raises error."""
    with pytest.raises(ValueError, match="Number of products must be at least 1."):
        MultiProductDiffusionModel(n_products=0)

    with pytest.raises(ValueError, match="Number of products must be at least 1."):
        MultiProductDiffusionModel(n_products=-1)


def test_multi_product_diffusion_model_init_with_params():
    """Test initializing with parameters."""
    p = [0.01, 0.02]
    Q = [[1.0, 0.1], [0.2, 1.0]]
    m = [100, 200]
    names = ["ProductA", "ProductB"]

    model = MultiProductDiffusionModel(n_products=2, p=p, Q=Q, m=m, names=names)

    assert model.n_products == 2
    assert model.names == names


def test_multi_product_diffusion_model_init_inconsistent_dims():
    """Test initializing with inconsistent parameter dimensions raises error."""
    p = [0.01, 0.02]  # 2 products
    Q = [[1.0, 0.1, 0.05], [0.2, 1.0, 0.1]]  # 2x3 matrix (should be 2x2)
    m = [100, 200]  # 2 products

    with pytest.raises(ValueError, match="Dimensions of p, Q, and m must be consistent with n_products."):
        MultiProductDiffusionModel(n_products=2, p=p, Q=Q, m=m)


def test_multi_product_diffusion_model_init_inconsistent_names():
    """Test initializing with inconsistent number of names raises error."""
    with pytest.raises(ValueError, match="Number of names must match n_products."):
        MultiProductDiffusionModel(n_products=2, names=["OnlyOneName"])


def test_multi_product_param_names():
    """Test the param_names property."""
    model = MultiProductDiffusionModel(n_products=2)

    param_names = model.param_names
    expected_names = [
        "p1",
        "p2",  # Innovation coefficients for products 1 and 2
        "q1",
        "q2",  # Imitation coefficients for products 1 and 2
        "m1",
        "m2",  # Market potentials for products 1 and 2
        "alpha_1_2",
        "alpha_2_1",  # Interaction coefficients
    ]

    assert set(param_names) == set(expected_names)


def test_multi_product_param_names_with_covariates():
    """Test the param_names property with covariates."""
    model = MultiProductDiffusionModel(n_products=2, covariates=["price", "advertising"])

    param_names = model.param_names

    # Check that standard params are included
    assert "p1" in param_names
    assert "q1" in param_names
    assert "m1" in param_names
    assert "alpha_1_2" in param_names

    # Check that covariate-related params are included
    assert "beta_p1_price" in param_names
    assert "beta_q1_price" in param_names
    assert "beta_m1_price" in param_names
    assert "beta_alpha_1_2_price" in param_names
    assert "beta_p1_advertising" in param_names
    assert "beta_q1_advertising" in param_names
    assert "beta_m1_advertising" in param_names
    assert "beta_alpha_1_2_advertising" in param_names


def test_multi_product_initial_guesses():
    """Test initial guesses generation."""
    model = MultiProductDiffusionModel(n_products=2)

    t = [0, 1, 2, 3, 4]
    y = [10, 20, 30, 40, 50]

    guesses = model.initial_guesses(t, y)

    # Check that all expected parameters are in the guesses
    assert "p1" in guesses
    assert "p2" in guesses
    assert "q1" in guesses
    assert "q2" in guesses
    assert "m1" in guesses
    assert "m2" in guesses
    assert "alpha_1_2" in guesses
    assert "alpha_2_1" in guesses

    # Check default values
    assert guesses["p1"] == 0.001
    assert guesses["q1"] == 0.1
    assert guesses["m1"] == 25.0  # max_y / n_products = 50 / 2 = 25.0
    assert guesses["m2"] == 25.0  # max_y / n_products = 50 / 2 = 25.0


def test_multi_product_initial_guesses_with_covariates():
    """Test initial guesses generation with covariates."""
    model = MultiProductDiffusionModel(n_products=2, covariates=["price"])

    t = [0, 1, 2, 3, 4]
    y = [10, 20, 30, 40, 50]

    guesses = model.initial_guesses(t, y)

    # Check that covariate-related guesses are included
    assert "beta_p1_price" in guesses
    assert "beta_q1_price" in guesses
    assert "beta_m1_price" in guesses
    assert "beta_alpha_1_2_price" in guesses

    # Check default value for covariate effects
    assert guesses["beta_p1_price"] == 0.0


def test_multi_product_bounds():
    """Test bounds generation."""
    model = MultiProductDiffusionModel(n_products=2)

    t = [0, 1, 2, 3, 4]
    y = [10, 20, 30, 40, 50]

    bounds = model.bounds(t, y)

    # Check bounds for standard parameters
    assert "p1" in bounds
    assert bounds["p1"] == (1e-6, 0.1)
    assert bounds["q1"] == (1e-6, 1.0)
    assert bounds["m1"] == (0, 100)  # max_y * 2 = 50 * 2 = 100

    # Check alpha bounds
    assert "alpha_1_2" in bounds
    assert bounds["alpha_1_2"] == (0, 2.0)


def test_multi_product_bounds_with_covariates():
    """Test bounds generation with covariates."""
    model = MultiProductDiffusionModel(n_products=2, covariates=["price"])

    t = [0, 1, 2, 3, 4]
    y = [10, 20, 30, 40, 50]

    bounds = model.bounds(t, y)

    # Check bounds for covariate-related parameters
    assert "beta_p1_price" in bounds
    assert bounds["beta_p1_price"] == (-np.inf, np.inf)


def test_multi_product_predict_without_params():
    """Test that predict raises an error when parameters are not set."""
    model = MultiProductDiffusionModel(n_products=2)

    with pytest.raises(RuntimeError, match="Model parameters"):
        model.predict([0, 1, 2])


def test_multi_product_score_without_params():
    """Test that score raises an error when the model hasn't been fitted."""
    model = MultiProductDiffusionModel(n_products=2)

    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.score([0, 1, 2], [0, 1, 2])


def test_multi_product_predict_adoption_rate_without_params():
    """Test that predict_adoption_rate raises an error when the model hasn't been fitted."""
    model = MultiProductDiffusionModel(n_products=2)

    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict_adoption_rate([0, 1, 2])
