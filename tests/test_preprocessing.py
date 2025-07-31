import pytest
import pandas as pd
import numpy as np
from innovate.utils.preprocessing import apply_rolling_average, apply_sarima


@pytest.fixture
def synthetic_time_series_data():
    dates = pd.date_range(start="2020-01-01", periods=120, freq="M")
    data = pd.Series(np.linspace(10, 1000, 120), index=dates)
    return data


def test_apply_rolling_average(synthetic_time_series_data):
    data = synthetic_time_series_data
    smoothed_data = apply_rolling_average(data, window=3)

    assert smoothed_data is not None
    assert len(smoothed_data) == len(data)
    assert smoothed_data.isnull().sum() == 2  # First two values will be NaN


def test_apply_sarima(synthetic_time_series_data):
    data = synthetic_time_series_data

    # These orders are just for testing purposes
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    fitted_values = apply_sarima(data, order=order, seasonal_order=seasonal_order)

    assert fitted_values is not None
    assert len(fitted_values) == len(data)
