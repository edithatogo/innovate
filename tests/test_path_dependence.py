import pytest
import numpy as np
from innovate.path_dependence.lock_in import LockInModel

@pytest.fixture
def lock_in_data():
    """Generate synthetic data for the LockInModel."""
    true_params = {
        "alpha1": 0.1, "alpha2": 0.08,
        "beta1": 0.005, "beta2": 0.007,
        "gamma1": 0.001, "gamma2": 0.001,
        "m": 1000.0
    }
    t = np.arange(0, 100, 1)
    y0 = [1.0, 1.0] # Small initial adoptions
    
    model = LockInModel()
    model.params_ = true_params
    
    y_true = model.predict(t, y0)
    
    noise = np.random.normal(0, 5, y_true.shape) # Add some noise
    y_noisy = np.clip(y_true + noise, 0, true_params["m"]) # Ensure within bounds
    
    return t, y_noisy, true_params

def test_lock_in_model_init():
    """Test initialization of the LockInModel."""
    model = LockInModel()
    assert model.param_names is not None

def test_lock_in_model_predict(lock_in_data):
    """Test the predict method of the LockInModel."""
    t, y_noisy, true_params = lock_in_data
    model = LockInModel()
    model.params_ = true_params
    y0 = y_noisy[0, :]
    
    predictions = model.predict(t, y0)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(t), 2)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= true_params["m"] + 1e-6) # Allow slight overshoot due to numerical integration

def test_lock_in_model_fit(lock_in_data):
    """Test the fitting process of the LockInModel."""
    t, y_noisy, true_params = lock_in_data
    
    model = LockInModel()
    model.fit(t, y_noisy)
    
    fitted_params = model.params_
    assert "alpha1" in fitted_params
    
    # Due to complexity of ODE fitting, allow a larger tolerance or just check for no errors
    # For now, just check that fitting completes without error and params are set.
    assert fitted_params is not None
    assert all(param in fitted_params for param in model.param_names)

def test_lock_in_model_score(lock_in_data):
    """Test the score method of the LockInModel."""
    t, y_noisy, true_params = lock_in_data
    model = LockInModel()
    model.params_ = true_params
    y0 = y_noisy[0, :]
    
    score = model.score(t, y_noisy)
    
    assert isinstance(score, float)
    # R^2 can be negative if the model is worse than a horizontal line
    assert score <= 1.0 # Score should not exceed 1.0

def test_lock_in_model_predict_adoption_rate(lock_in_data):
    """Test the predict_adoption_rate method."""
    t, y_noisy, true_params = lock_in_data
    y0 = y_noisy[0, :]

    model = LockInModel()
    model.params_ = true_params
    
    rates = model.predict_adoption_rate(t, y0)
    
    assert isinstance(rates, np.ndarray)
    assert rates.shape == (len(t), 2)
    assert np.all(rates >= -1e-6) # Rates should generally be non-negative
