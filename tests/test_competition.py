# tests/test_competition.py

import pytest
import numpy as np
import pandas as pd
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
    assert model.N == 2
    assert len(model.p) == 2
    assert model.Q.shape == (2, 2)
    assert len(model.m) == 2
    assert model.names == product_names

def test_competition_model_init_empty():
    """Test that initialization with invalid parameters raises an error."""
    with pytest.raises(ValueError, match="Dimensions of p, Q, and m must be consistent."):
        MultiProductDiffusionModel(p=[], Q=[], m=[])

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