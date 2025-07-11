# tests/test_bayesian_fitter.py

import pytest
import numpy as np
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.bayesian_fitter import BayesianFitter

@pytest.fixture
def synthetic_logistic_data():
    t = np.linspace(0, 20, 100)
    # True parameters: L=1.0, k=1.5, x0=10.0
    y = 1.0 / (1 + np.exp(-1.5 * (t - 10.0))) + np.random.normal(0, 0.01, len(t))
    return t, y

def test_bayesian_fitter(synthetic_logistic_data):
    t, y = synthetic_logistic_data
    model = LogisticModel()
    fitter = BayesianFitter(model, draws=1000, tune=1000, chains=2)
    fitter.fit(t, y)

    assert model.params_ is not None
    assert len(model.params_) == 3 # L, k, x0
    # Allow a larger tolerance for Bayesian fitting
    assert np.allclose(list(model.params_.values()), [1.0, 1.5, 10.0], atol=0.5)
