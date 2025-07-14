import pytest
import numpy as np
from innovate.dynamics.growth import SymmetricGrowth, SkewedGrowth, DualInfluenceGrowth

def test_symmetric_growth():
    model = SymmetricGrowth()
    t = np.linspace(0, 50, 100)
    y = model.predict_cumulative(t, 1, 1000)
    assert len(y) == 100
    assert y[0] == 1
    assert y[-1] < 1000

def test_skewed_growth():
    model = SkewedGrowth()
    t = np.linspace(0, 50, 100)
    y = model.predict_cumulative(t, 1, 1000)
    assert len(y) == 100
    assert y[0] > 0
    assert y[-1] < 1000

def test_dual_influence_growth():
    model = DualInfluenceGrowth()
    t = np.linspace(0, 50, 100)
    y = model.predict_cumulative(t, 1, 1000)
    assert len(y) == 100
    assert y[0] == 1
    assert y[-1] < 1000
