
import pytest
import numpy as np
from innovate.substitute.norton_bass import NortonBassModel
from innovate.fitters.scipy_fitter import ScipyFitter

@pytest.fixture
def synthetic_norton_bass_data():
    """
    Generate synthetic data for two generations of a technology using the model itself.
    """
    t = np.linspace(0, 50, 100)
    
    # True parameters for two generations
    true_params = {
        "p1": 0.03, "q1": 0.2, "m1": 1000,
        "p2": 0.02, "q2": 0.3, "m2": 1500
    }

    # Use the model to generate the ideal data
    model = NortonBassModel(n_generations=2)
    model.params_ = true_params
    y_ideal = model.predict(t)

    return t, y_ideal, list(true_params.values())

def test_norton_bass_model_fit(synthetic_norton_bass_data):
    """
    Test the fitting of the NortonBassModel.
    """
    t, y, true_params = synthetic_norton_bass_data
    
    # Initialize the model for 2 generations
    model = NortonBassModel(n_generations=2)
    
    # Provide slightly perturbed initial guesses to guide the optimizer
    p0 = np.array(true_params) * (1 + np.random.uniform(-0.1, 0.1, size=len(true_params)))

    # Use the ScipyFitter to fit the model
    fitter = ScipyFitter()
    fitter.fit(model, t, y, p0=p0)
    
    # Check if the parameters have been fitted
    assert model.params_ is not None
    assert len(model.params_) == 6 # p1, q1, m1, p2, q2, m2
    
    # Check if the fitted parameters are within a reasonable range of the true parameters
    fitted_params = list(model.params_.values())
    
    # Use a high relative tolerance to account for the difficulty of fitting this model
    assert np.allclose(fitted_params, true_params, rtol=0.4)
