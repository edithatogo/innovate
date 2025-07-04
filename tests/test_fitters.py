import pytest
from heartflow.fitters.scipy_fitter import ScipyFitter
from heartflow.models.logistic import LogisticModel
import numpy as np

def test_scipy_fitter():
    # Generate some synthetic data
    t = np.linspace(0, 20, 100)
    y = 1 / (1 + np.exp(-1.5 * (t - 10)))

    model = LogisticModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, y)

    assert fitter.params_ is not None
    assert len(fitter.params_) == 3 # L, k, t0
    assert np.allclose(list(fitter.params_.values()), [1.0, 1.5, 10.0], atol=0.1)
