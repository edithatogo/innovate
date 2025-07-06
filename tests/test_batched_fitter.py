import pytest
import numpy as np
from heartflow.fitters.batched_fitter import BatchedFitter
from heartflow.fitters.scipy_fitter import ScipyFitter
from heartflow.models.logistic import LogisticModel

@pytest.fixture
def synthetic_batched_data():
    # Create two different logistic curves
    t1 = np.linspace(0, 20, 50)
    y1 = 1.0 / (1 + np.exp(-1.5 * (t1 - 10.0))) + np.random.normal(0, 0.01, len(t1))

    t2 = np.linspace(0, 30, 50)
    y2 = 1.5 / (1 + np.exp(-0.5 * (t2 - 15.0))) + np.random.normal(0, 0.01, len(t2))
    
    return [t1, t2], [y1, y2]

def test_batched_fitter_fit(synthetic_batched_data):
    t_batched, y_batched = synthetic_batched_data
    
    model = LogisticModel()
    fitter = ScipyFitter()
    batched_fitter = BatchedFitter(model, fitter)
    
    fitted_params = batched_fitter.fit(t_batched, y_batched)
    
    assert fitted_params is not None
    assert fitted_params.shape == (2, 3) # 2 datasets, 3 parameters
    
    # Check that the fitted parameters are close to the true parameters
    assert np.allclose(fitted_params[0], [1.0, 1.5, 10.0], atol=0.2)
    assert np.allclose(fitted_params[1], [1.5, 0.5, 15.0], atol=0.2)

def test_batched_fitter_predict(synthetic_batched_data):
    t_batched, y_batched = synthetic_batched_data
    
    model = LogisticModel()
    fitter = ScipyFitter()
    batched_fitter = BatchedFitter(model, fitter)
    
    batched_fitter.fit(t_batched, y_batched)
    
    predictions = batched_fitter.predict(t_batched)
    
    assert predictions is not None
    assert predictions.shape == (2, 50)

def test_batched_fitter_jax(synthetic_batched_data):
    from heartflow.backend import use_backend, current_backend
    from heartflow.fitters.jax_fitter import JaxFitter
    original_backend = current_backend
    use_backend("jax")

    t_batched, y_batched = synthetic_batched_data
    
    model = LogisticModel()
    fitter = JaxFitter()
    batched_fitter = BatchedFitter(model, fitter)
    
    fitted_params = batched_fitter.fit(t_batched, y_batched)
    
    assert fitted_params is not None
    assert fitted_params.shape == (2, 3)
    
    # Check that the fitted parameters are close to the true parameters
    assert np.allclose(fitted_params[0], [1.0, 1.5, 10.0], atol=0.2)
    assert np.allclose(fitted_params[1], [1.5, 0.5, 15.0], atol=0.2)

    predictions = batched_fitter.predict(t_batched)
    
    assert predictions is not None
    assert predictions.shape == (2, 50)

    use_backend(original_backend.__class__.__name__.lower().replace('backend', ''))
