import pytest
import numpy as np
import pandas as pd
from innovate.models.bass import BassModel
from innovate.models.gompertz import GompertzModel
from innovate.models.logistic import LogisticModel
from innovate.models.competition import MultiProductDiffusionModel
from innovate.fitters.scipy_fitter import ScipyFitter

# Fixture for common test data
@pytest.fixture
def synthetic_data():
    time_points = np.arange(1, 51)
    # True parameters for Bass model
    true_p = 0.03
    true_q = 0.3
    true_m = 1000

    def _bass_cumulative_true(t, p, q, m):
        exp_term = np.exp(-(p + q) * t)
        return m * (1 - exp_term) / (1 + (q / p) * exp_term)

    cumulative_adoptions = _bass_cumulative_true(time_points, true_p, true_q, true_m)
    
    # Add some noise and ensure non-negativity and cumulativity
    np.random.seed(42)
    noise = np.random.normal(0, 10, len(time_points))
    observed_adoptions = cumulative_adoptions + noise
    observed_adoptions[observed_adoptions < 0] = 0
    observed_adoptions = np.maximum.accumulate(observed_adoptions)

    return time_points, observed_adoptions

# Test BassModel
def test_bass_model_fit_predict(synthetic_data):
    t, y = synthetic_data
    model = BassModel()
    model.fit(t, y)

    assert model.params_ is not None
    assert all(param in model.params_ for param in ["p", "q", "m"])
    assert model.params_["m"] > 0

    predictions = model.predict(t)
    assert len(predictions) == len(t)
    assert np.all(predictions >= 0)
    assert np.all(np.diff(predictions) >= -1e-6) # Ensure non-decreasing (allowing for small numerical errors)

def test_bass_model_score(synthetic_data):
    t, y = synthetic_data
    model = BassModel()
    model.fit(t, y)
    score = model.score(t, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0 # R^2 can be negative, but for good fits, it should be positive

# Test GompertzModel
def test_gompertz_model_fit_predict(synthetic_data):
    t, y = synthetic_data
    model = GompertzModel()
    model.fit(t, y)

    assert model.params_ is not None
    assert all(param in model.params_ for param in ["a", "b", "c"])
    assert model.params_["a"] > 0

    predictions = model.predict(t)
    assert len(predictions) == len(t)
    assert np.all(predictions >= 0)
    assert np.all(np.diff(predictions) >= -1e-6)

def test_gompertz_model_score(synthetic_data):
    t, y = synthetic_data
    model = GompertzModel()
    model.fit(t, y)
    score = model.score(t, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

# Test LogisticModel
def test_logistic_model_fit_predict(synthetic_data):
    t, y = synthetic_data
    model = LogisticModel()
    model.fit(t, y)

    assert model.params_ is not None
    assert all(param in model.params_ for param in ["L", "k", "x0"])
    assert model.params_["L"] > 0

    predictions = model.predict(t)
    assert len(predictions) == len(t)
    assert np.all(predictions >= 0)
    assert np.all(np.diff(predictions) >= -1e-6)

def test_logistic_model_score(synthetic_data):
    t, y = synthetic_data
    model = LogisticModel()
    model.fit(t, y)
    score = model.score(t, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

# Test MultiProductDiffusionModel
def test_multi_product_model_predict():
    p_vals = [0.02, 0.015]
    Q_matrix = [
        [0.3, 0.05],
        [0.03, 0.25]
    ]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]

    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    time_horizon = np.arange(1, 101)
    predictions_df = model.predict(time_horizon)

    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == len(time_horizon)
    assert list(predictions_df.columns) == product_names
    assert np.all(predictions_df.values >= 0)
    # Check if cumulative (each product's adoption should be non-decreasing)
    for col in product_names:
        assert np.all(np.diff(predictions_df[col].values) >= -1e-6)

def test_multi_product_model_score():
    p_vals = [0.02, 0.015]
    Q_matrix = [
        [0.3, 0.05],
        [0.03, 0.25]
    ]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]

    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    time_horizon = np.arange(1, 101)
    predictions_df = model.predict(time_horizon)

    # Create dummy observed data for scoring
    observed_df = predictions_df * (1 + np.random.normal(0, 0.05, predictions_df.shape))
    observed_df[observed_df < 0] = 0
    # Ensure observed data is also cumulative for a fair comparison
    for col in observed_df.columns:
        observed_df[col] = np.maximum.accumulate(observed_df[col])

    score = model.score(time_horizon, observed_df)
    assert isinstance(score, float)
    # R^2 can be negative if the model is worse than a horizontal line
    # For a model predicting its own slightly noisy version, it should be high
    assert score > 0.5 # Expect a reasonably good score

# Test ScipyFitter
def test_scipy_fitter_single_model(synthetic_data):
    t, y = synthetic_data
    fitter = ScipyFitter()
    model = BassModel()
    
    # The ScipyFitter for Phase 1 just calls the model's fit method
    fitter.fit(model, t, y)

    assert fitter.params_ is not None
    assert all(param in fitter.params_ for param in ["p", "q", "m"])
    assert model.params_ == fitter.params_ # Ensure model's internal params are set

def test_scipy_fitter_multi_product_model_not_implemented():
    fitter = ScipyFitter()
    p_vals = [0.02, 0.015]
    Q_matrix = [
        [0.3, 0.05],
        [0.03, 0.25]
    ]
    m_vals = [1000, 800]
    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals)
    time_points = np.arange(1, 10)
    data = pd.DataFrame(np.random.rand(len(time_points), 2) * 100, columns=model.names)
    
    with pytest.raises(NotImplementedError, match="Fitting MultiProductDiffusionModel with ScipyFitter is not yet implemented"):
        fitter.fit(model, time_points, data)
