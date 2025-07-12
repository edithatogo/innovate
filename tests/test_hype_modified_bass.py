# tests/test_hype_modified_bass.py

import pytest
import numpy as np
from innovate.hype.hype_modified_bass import HypeModifiedBassModel
from innovate.diffuse.bass import BassModel
from innovate.hype.hype_cycle import HypeCycleModel

@pytest.fixture
def fitted_bass_model():
    """A fitted Bass model."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1.0}
    return model

@pytest.fixture
def hype_cycle_model():
    """A Hype Cycle model with typical parameters."""
    model = HypeCycleModel()
    model.params_ = {
        "k": 0.1,
        "t0": 50,
        "a_hype": 1.0,
        "t_hype": 20,
        "w_hype": 8,
        "a_d": 0.9,
        "t_d": 40,
        "w_d": 15,
    }
    return model

def test_hype_modified_bass_init(fitted_bass_model, hype_cycle_model):
    """Test initialization of the HypeModifiedBassModel."""
    model = HypeModifiedBassModel(bass_model=fitted_bass_model, hype_model=hype_cycle_model)
    assert model.bass_model is not None
    assert model.hype_model is not None

def test_hype_modified_bass_predict(fitted_bass_model, hype_cycle_model):
    """Test the predict method of the HypeModifiedBassModel."""
    model = HypeModifiedBassModel(bass_model=fitted_bass_model, hype_model=hype_cycle_model)
    
    t = np.arange(0, 100, 1)
    y0 = 0.01
    
    adoption = model.predict(t, y0)
    
    assert isinstance(adoption, np.ndarray)
    assert adoption.shape == (len(t),)
    assert np.all(adoption >= 0)
    # The adoption should be generally increasing
    assert adoption[-1] > adoption[0]

def test_predict_without_params(fitted_bass_model, hype_cycle_model):
    """Test that predict raises an error if params are not set."""
    # Test with no params on hype model
    bass_model = fitted_bass_model
    hype_model_no_params = HypeCycleModel()
    model = HypeModifiedBassModel(bass_model=bass_model, hype_model=hype_model_no_params)
    with pytest.raises(RuntimeError):
        model.predict(np.arange(10), 0.01)

    # Test with no params on bass model
    bass_model_no_params = BassModel()
    hype_model = hype_cycle_model
    model = HypeModifiedBassModel(bass_model=bass_model_no_params, hype_model=hype_model)
    with pytest.raises(RuntimeError):
        model.predict(np.arange(10), 0.01)