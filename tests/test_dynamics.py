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
    # closed‐form initial value: K * exp(-b) with default b=1.0
    expected_initial = 1000 * np.exp(-1.0)
    assert y[0] == pytest.approx(expected_initial)
    assert y[-1] < 1000


def test_dual_influence_growth():
    model = DualInfluenceGrowth()
    t = np.linspace(0, 50, 100)
    y = model.predict_cumulative(t, 1, 1000)
    assert len(y) == 100
    assert y[0] == 1
    assert y[-1] < 1000
