# tests/test_bayesian_fitter.py

import pytest
import numpy as np
from innovate.diffuse.bass import BassModel
from innovate.fitters.bayesian_fitter import BayesianFitter

@pytest.fixture
def synthetic_bass_data():
    np.random.seed(0)
    t = np.linspace(0, 20, 100)
    p, q, m = 0.03, 0.38, 1.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 0.01, len(t))
    return t, y

def test_bayesian_fitter(synthetic_bass_data):
    t, y = synthetic_bass_data
    model = BassModel()
    fitter = BayesianFitter(model, draws=20, tune=20, chains=1, cores=1)
    fitter.fit(t, y, target_accept=0.9)

    assert model.params_ is not None
    assert len(model.params_) == 3  # p, q, m
    assert np.allclose(
        list(model.params_.values()), [0.03, 0.38, 1.0], atol=0.1
    )

    # Test get_parameter_estimates
    estimates = fitter.get_parameter_estimates()
    assert isinstance(estimates, dict)
    assert len(estimates) == 3
    assert "p" in estimates
    assert "q" in estimates
    assert "m" in estimates

    # Test get_confidence_intervals
    intervals = fitter.get_confidence_intervals()
    assert isinstance(intervals, dict)
    assert len(intervals) == 3
    assert set(intervals.keys()) == {"p", "q", "m"}

    # Test get_summary
    summary = fitter.get_summary()
    assert summary is not None
