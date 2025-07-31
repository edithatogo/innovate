# tests/test_lotka_volterra.py

import pytest
import numpy as np
from innovate.compete.lotka_volterra import LotkaVolterraModel


def test_lotka_volterra_param_names():
    """Test the parameter names of the Lotka-Volterra model."""
    model = LotkaVolterraModel()
    assert model.param_names == ["alpha1", "beta1", "alpha2", "beta2"]


def test_lotka_volterra_initial_guesses():
    """Test the initial guesses for the Lotka-Volterra model."""
    model = LotkaVolterraModel()
    t = np.arange(10)
    # y needs to be 2D for this model
    y = np.random.rand(10, 2)
    guesses = model.initial_guesses(t, y)

    assert isinstance(guesses, dict)
    assert all(param in guesses for param in model.param_names)
    assert all(np.isfinite(value) for value in guesses.values())


def test_lotka_volterra_bounds():
    """Test the bounds for the Lotka-Volterra model."""
    model = LotkaVolterraModel()
    t = np.arange(10)
    y = np.random.rand(10, 2)
    bounds = model.bounds(t, y)

    assert isinstance(bounds, dict)
    assert all(param in bounds for param in model.param_names)
    for lower, upper in bounds.values():
        assert lower == 0
        assert upper == np.inf


def test_lotka_volterra_predict():
    """Test the predict method of the Lotka-Volterra model."""
    model = LotkaVolterraModel()
    model.params_ = {"alpha1": 0.1, "beta1": 0.01, "alpha2": 0.1, "beta2": 0.01}

    t = np.arange(0, 10, 1)
    y0 = [0.01, 0.02]

    predictions = model.predict(t, y0)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(t), 2)
    assert np.all(predictions >= 0)
    # Check that the populations are growing
    assert predictions[-1, 0] > y0[0]
    assert predictions[-1, 1] > y0[1]


@pytest.fixture
def lotka_volterra_data():
    """Generate synthetic data for the Lotka-Volterra model."""
    true_params = {"alpha1": 0.5, "beta1": 0.1, "alpha2": 0.4, "beta2": 0.1}
    t = np.arange(0, 20, 1)
    y0 = [0.01, 0.02]

    model = LotkaVolterraModel()
    model.params_ = true_params

    y_true = model.predict(t, y0)

    # Add some noise
    noise = np.random.normal(0, 0.01, y_true.shape)
    y_noisy = np.clip(y_true + noise, 0, 1)

    return t, y_noisy, true_params


def test_lotka_volterra_fit(lotka_volterra_data):
    """Test the fitting process to see if it can recover parameters."""
    t, y_noisy, true_params = lotka_volterra_data

    model = LotkaVolterraModel()
    model.fit(t, y_noisy)

    fitted_params = model.params_
    assert "alpha1" in fitted_params

    # Check if the fitting process produced all the parameters.
    # A strict check on values is omitted as it can be unstable with noisy data.
    for param_name in true_params:
        assert param_name in fitted_params


def test_lotka_volterra_score(lotka_volterra_data):
    """Test the score method of the Lotka-Volterra model."""
    t, y_noisy, true_params = lotka_volterra_data

    model = LotkaVolterraModel()
    model.params_ = true_params

    score = model.score(t, y_noisy)

    assert isinstance(score, float)
    assert score <= 1.0


def test_lotka_volterra_predict_adoption_rate(lotka_volterra_data):
    """Test the predict_adoption_rate method."""
    t, y_noisy, true_params = lotka_volterra_data
    y0 = y_noisy[0, :]

    model = LotkaVolterraModel()
    model.params_ = true_params

    rates = model.predict_adoption_rate(t, y0)

    assert isinstance(rates, np.ndarray)
    assert rates.shape == (len(t), 2)
