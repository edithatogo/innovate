# tests/test_preprocess.py

import pytest
import pandas as pd
import numpy as np
from innovate.preprocess import stl_decomposition

@pytest.fixture
def seasonal_time_series():
    """Generate a synthetic time series with a clear seasonal pattern."""
    dates = pd.date_range(start='2020-01-01', periods=120, freq='M')
    trend = np.linspace(10, 100, 120)
    seasonal = 10 * np.sin(np.linspace(0, 10 * 2 * np.pi, 120)) # 10 years of monthly data
    noise = np.random.normal(0, 1, 120)
    return pd.Series(trend + seasonal + noise, index=dates)

def test_stl_decomposition(seasonal_time_series):
    """Test the stl_decomposition function."""
    decomposed = stl_decomposition(seasonal_time_series, period=12)
    
    assert isinstance(decomposed, pd.DataFrame)
    assert all(col in decomposed.columns for col in ['trend', 'seasonal', 'residual'])
    assert not decomposed.isnull().values.any()
    
    # Check if the sum of components is close to the original series
    reconstructed = decomposed['trend'] + decomposed['seasonal'] + decomposed['residual']
    pd.testing.assert_series_equal(seasonal_time_series, reconstructed, check_names=False)

def test_stl_decomposition_no_datetime_index():
    """Test that stl_decomposition raises an error for non-DatetimeIndex."""
    series = pd.Series(np.random.rand(100))
    with pytest.raises(TypeError, match="The input series must have a DatetimeIndex."):
        stl_decomposition(series, period=12)
