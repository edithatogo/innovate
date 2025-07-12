# tests/test_complementary_goods.py

import pytest
import numpy as np
from innovate.ecosystem.complementary_goods import ComplementaryGoodsModel

@pytest.fixture
def complementary_goods_data():
    """Generate synthetic data for the ComplementaryGoodsModel."""
    true_params = {"k1": 0.3, "k2": 0.2, "c1": 0.1, "c2": 0.15}
    t = np.arange(0, 20, 1)
    y0 = [0.01, 0.01]
    
    model = ComplementaryGoodsModel()
    model.params_ = true_params
    
    y_true = model.predict(t, y0)
    
    noise = np.random.normal(0, 0.01, y_true.shape)
    y_noisy = np.clip(y_true + noise, 0, 1)
    
    return t, y_noisy, true_params

def test_complementary_goods_model_init():
    """Test initialization of the ComplementaryGoodsModel."""
    model = ComplementaryGoodsModel()
    assert model.param_names is not None

def test_complementary_goods_model_predict(complementary_goods_data):
    """Test the predict method of the ComplementaryGoodsModel."""
    t, y_noisy, true_params = complementary_goods_data
    model = ComplementaryGoodsModel()
    model.params_ = true_params
    
    predictions = model.predict(t, y_noisy[0, :])
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(t), 2)
    assert np.all(predictions >= 0)

def test_complementary_goods_model_fit(complementary_goods_data):
    """Test the fitting process of the ComplementaryGoodsModel."""
    t, y_noisy, true_params = complementary_goods_data
    
    model = ComplementaryGoodsModel()
    model.fit(t, y_noisy)
    
    fitted_params = model.params_
    
    for param_name in true_params:
        assert np.isclose(fitted_params[param_name], true_params[param_name], rtol=1.0)

def test_complementary_goods_model_score(complementary_goods_data):
    """Test the score method of the ComplementaryGoodsModel."""
    t, y_noisy, true_params = complementary_goods_data
    model = ComplementaryGoodsModel()
    model.params_ = true_params
    
    score = model.score(t, y_noisy)
    
    assert isinstance(score, float)

def test_complementary_goods_model_predict_adoption_rate(complementary_goods_data):
    """Test the predict_adoption_rate method."""
    t, y_noisy, true_params = complementary_goods_data
    model = ComplementaryGoodsModel()
    model.params_ = true_params
    
    rates = model.predict_adoption_rate(t, y_noisy[0, :])
    
    assert isinstance(rates, np.ndarray)
    assert rates.shape == (len(t), 2)