# tests/test_competition.py

import pytest
import numpy as np
import pandas as pd
from typing import Sequence
from innovate.compete.competition import MultiProductDiffusionModel
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.diffuse.bass import BassModel

@pytest.fixture
def fitted_bass_model():
    """A fitted Bass model."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1.0}
    return model

@pytest.fixture
def fitted_fisher_pry_model():
    """A fitted Fisher-Pry model."""
    model = FisherPryModel()
    model.params_ = {"alpha": 0.5, "t0": 10}
    return model

def test_competition_model_init():
    """Test initialization of the MultiProductDiffusionModel."""
    p_vals = [0.02, 0.015]
    Q_matrix = [[0.3, 0.05], [0.03, 0.25]]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]
    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    assert len(model.p) == 2
    assert model.Q.shape == (2, 2)
    assert len(model.m) == 2
    assert model.names == product_names

def test_competition_model_init_empty():
    """Test that initialization with invalid parameters raises an error."""
    with pytest.raises(ValueError):
        MultiProductDiffusionModel(p=[0.1], Q=[[0.1, 0.2]], m=[100, 200])

def test_multi_product_model_fit_basic():
    """Test the fit method of the MultiProductDiffusionModel with basic parameters."""
    # Generate synthetic data for two products
    t = np.arange(1, 51)
    # Simulate Bass-like diffusion for two products with some interaction
    # These are not exact, just for generating plausible data
    y1_true = 1000 / (1 + np.exp(-0.3 * (t - 20))) # Logistic-like
    y2_true = 800 / (1 + np.exp(-0.25 * (t - 25))) # Logistic-like

    # Add some noise and ensure cumulative
    np.random.seed(42)
    y1_obs = np.maximum.accumulate(y1_true + np.random.normal(0, 20, len(t)))
    y2_obs = np.maximum.accumulate(y2_true + np.random.normal(0, 15, len(t)))
    y_obs = np.vstack([y1_obs, y2_obs]).T

    init_p = [0.01, 0.01]
    init_Q = [[0.1, 0.01], [0.01, 0.1]]
    init_m = [1200, 1000]
    model = MultiProductDiffusionModel(p=init_p, Q=init_Q, m=init_m, names=["ProdA", "ProdB"])
    model.fit(t, y_obs)

    assert model.params_ is not None
    assert len(model.params_) == len(model.param_names)

    predictions = model.predict(t)
    assert predictions.shape == y_obs.shape
    assert np.all(predictions >= 0)

    df_obs = pd.DataFrame(y_obs, index=t, columns=["ProdA", "ProdB"])
    score = model.score(t, df_obs)
    assert score > 0.5 # Should be a reasonably good fit

def test_multi_product_model_predict_basic():
    """Test the predict method of the MultiProductDiffusionModel with basic parameters."""
    p_vals = [0.02, 0.015]
    Q_matrix = [[0.3, 0.05], [0.03, 0.25]]
    m_vals = [1000, 800]
    product_names = ["ProdA", "ProdB"]

    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)
    
    t = np.arange(1, 101) # Use a longer time horizon for more meaningful prediction
    predictions_df = model.predict(t)
    
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == len(t)
    assert list(predictions_df.columns) == product_names
    assert np.all(predictions_df.values >= 0)
    # Check if cumulative (each product's adoption should be non-decreasing)
    for col in product_names:
        assert np.all(np.diff(predictions_df[col].values) >= -1e-6)