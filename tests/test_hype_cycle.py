# tests/test_hype_cycle.py

import pytest
import numpy as np
from innovate.hype.hype_cycle import HypeCycleModel

@pytest.fixture
def hype_cycle_params():
    """Parameters for a typical Hype Cycle."""
    return {
        "k": 0.1,
        "t0": 50,
        "a_hype": 1.0,
        "t_hype": 20,
        "w_hype": 8,
        "a_d": 0.9,
        "t_d": 40,
        "w_d": 15,
    }

def test_hype_cycle_model_init():
    """Test initialization of the HypeCycleModel."""
    model = HypeCycleModel()
    assert model.param_names is not None

def test_hype_cycle_model_predict(hype_cycle_params):
    """Test the predict method of the HypeCycleModel."""
    model = HypeCycleModel()
    model.params_ = hype_cycle_params
    
    t = np.arange(0, 100, 1)
    visibility = model.predict(t)
    
    assert isinstance(visibility, np.ndarray)
    assert visibility.shape == (len(t),)
    assert np.all(visibility >= 0)

    # Check for the peak of inflated expectations
    peak_time = t[np.argmax(visibility)]

    # Check for the trough of disillusionment
    # Find the minimum after the peak
    trough_time = t[np.argmin(visibility[peak_time:]) + peak_time]

    assert peak_time < trough_time
    assert visibility[peak_time] > visibility[trough_time]

def test_predict_without_params():
    """Test that predict raises an error if params are not set."""
    model = HypeCycleModel()
    with pytest.raises(RuntimeError):
        model.predict(np.arange(10))