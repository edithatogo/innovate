# tests/test_fisher_pry.py

import pytest
import numpy as np
import pandas as pd
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.diffuse.fitters.scipy_fitter import ScipyFitter

@pytest.fixture
def fisher_pry_data():
    """Generate synthetic data for a Fisher-Pry model."""
    alpha = 0.5
    t0 = 10
    t = np.arange(0, 25, 1)
    # Market share fraction (0 to 1)
    y = 1 / (1 + np.exp(-alpha * (t - t0)))
    # Add some noise
    noise = np.random.normal(0, 0.02, len(t))
    y_noisy = np.clip(y + noise, 0, 1)
    return t, y_noisy, {"alpha": alpha, "t0": t0}

def test_fisher_pry_model_predict(fisher_pry_data):
    """Test the predict method with known parameters."""
    t, _, params = fisher_pry_data
    model = FisherPryModel()
    model.params_ = params
    
    predictions = model.predict(t)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(t)
    # The prediction should be close to the noise-free data
    y_true = 1 / (1 + np.exp(-params["alpha"] * (t - params["t0"])))
    np.testing.assert_allclose(predictions, y_true, atol=1e-6)

def test_fisher_pry_model_fit(fisher_pry_data):
    """Test the fitting process to see if it can recover parameters."""
    t, y_noisy, true_params = fisher_pry_data
    
    model = FisherPryModel()
    fitter = ScipyFitter()
    
    model.fit(fitter, t, y_noisy)
    
    fitted_params = model.params_
    assert "alpha" in fitted_params
    assert "t0" in fitted_params
    
    # Check if the recovered parameters are reasonably close to the true ones
    assert np.isclose(fitted_params["alpha"], true_params["alpha"], rtol=0.2)
    assert np.isclose(fitted_params["t0"], true_params["t0"], rtol=0.2)

def test_predict_adoption_rate(fisher_pry_data):
    """Test the adoption rate prediction."""
    t, _, params = fisher_pry_data
    model = FisherPryModel()
    model.params_ = params
    
    rates = model.predict_adoption_rate(t)
    
    assert isinstance(rates, np.ndarray)
    assert len(rates) == len(t)
    
    # The peak of the adoption rate should be at t0
    peak_time_index = np.argmax(rates)
    assert np.isclose(t[peak_time_index], params["t0"], atol=1.0)

def test_initial_guesses(fisher_pry_data):
    """Test the initial guess logic."""
    t, y_noisy, _ = fisher_pry_data
    model = FisherPryModel()
    guesses = model.initial_guesses(t, y_noisy)
    
    assert "alpha" in guesses
    assert "t0" in guesses
    assert np.isfinite(guesses["alpha"])
    assert np.isfinite(guesses["t0"])

def test_initial_guesses_edge_cases():
    """Test initial guesses with edge-case data."""
    model = FisherPryModel()
    t = np.arange(10)
    
    # All zeros
    y_zeros = np.zeros(10)
    guesses_zeros = model.initial_guesses(t, y_zeros)
    assert np.isfinite(guesses_zeros["alpha"])
    assert np.isfinite(guesses_zeros["t0"])
    
    # All ones
    y_ones = np.ones(10)
    guesses_ones = model.initial_guesses(t, y_ones)
    assert np.isfinite(guesses_ones["alpha"])
    assert np.isfinite(guesses_ones["t0"])

def test_fit_with_pandas_arrow(fisher_pry_data):
    """Test fitting with a pandas DataFrame using the pyarrow backend."""
    t, y_noisy, true_params = fisher_pry_data
    
    df = pd.DataFrame({'time': t, 'market_share': y_noisy})
    df = df.convert_dtypes(dtype_backend='pyarrow')
    
    model = FisherPryModel()
    fitter = ScipyFitter()
    
    model.fit(fitter, df['time'], df['market_share'])
    
    fitted_params = model.params_
    assert "alpha" in fitted_params
    assert "t0" in fitted_params
    
    assert np.isclose(fitted_params["alpha"], true_params["alpha"], rtol=0.2)
    assert np.isclose(fitted_params["t0"], true_params["t0"], rtol=0.2)

def test_fit_non_logistic_data():
    """
    Test fitting with non-logistic data.
    It should either complete with finite parameters or raise a RuntimeError.
    """
    t = np.arange(20)
    y_random = np.random.rand(20)
    
    model = FisherPryModel()
    fitter = ScipyFitter()
    
    try:
        model.fit(fitter, t, y_random)
        # If it fits, parameters should be finite
        assert np.isfinite(model.params_["alpha"])
        assert np.isfinite(model.params_["t0"])
    except RuntimeError:
        # If it fails to fit, that's also acceptable for this test
        pass
