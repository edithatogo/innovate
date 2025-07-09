import pytest
from innovate.diffuse.fitters.scipy_fitter import ScipyFitter
from innovate.diffuse.fitters.bootstrap_fitter import BootstrapFitter
from innovate.diffuse.fitters.jax_fitter import JaxFitter
from innovate.diffuse.logistic import LogisticModel
import numpy as np

@pytest.fixture
def synthetic_logistic_data():
    t = np.linspace(0, 20, 100)
    # True parameters: L=1.0, k=1.5, x0=10.0
    y = 1.0 / (1 + np.exp(-1.5 * (t - 10.0))) + np.random.normal(0, 0.01, len(t))
    return t, y

def test_scipy_fitter(synthetic_logistic_data):
    t, y = synthetic_logistic_data
    model = LogisticModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, y)

    assert model.params_ is not None
    assert len(model.params_) == 3 # L, k, x0
    # Allow a larger tolerance for fitting noise-added data
    assert np.allclose(list(model.params_.values()), [1.0, 1.5, 10.0], atol=0.2)

def test_bootstrap_fitter(synthetic_logistic_data):
    t, y = synthetic_logistic_data
    model = LogisticModel()
    base_fitter = ScipyFitter() # Instantiate a base fitter
    bootstrap_fitter = BootstrapFitter(base_fitter, n_bootstraps=10) # Pass the base fitter
    bootstrap_fitter.fit(model, t, y)

    assert len(bootstrap_fitter.bootstrapped_params) > 0
    
    param_estimates = bootstrap_fitter.get_parameter_estimates()
    assert "L" in param_estimates
    assert "k" in param_estimates
    assert "x0" in param_estimates
    assert len(param_estimates["L"]) == len(bootstrap_fitter.bootstrapped_params)

    cis = bootstrap_fitter.get_confidence_intervals()
    assert "L" in cis
    assert "k" in cis
    assert "x0" in cis
    assert "lower" in cis["L"]
    assert "upper" in cis["L"]

    ses = bootstrap_fitter.get_standard_errors()
    assert "L" in ses
    assert "k" in ses
    assert "x0" in ses
    assert ses["L"] >= 0

def test_jax_fitter(synthetic_logistic_data):
    from innovate.backend import use_backend, current_backend
    original_backend = current_backend
    use_backend("jax")

    t, y = synthetic_logistic_data
    model = LogisticModel()
    jax_fitter = JaxFitter()
    jax_fitter.fit(model, t, y)

    assert model.params_ is not None
    assert len(model.params_) == 3 # L, k, x0
    # Allow a larger tolerance for JAX fitting
    assert np.allclose(list(model.params_.values()), [1.0, 1.5, 10.0], atol=0.2)
    
    use_backend(original_backend.__class__.__name__.lower().replace('backend', ''))

