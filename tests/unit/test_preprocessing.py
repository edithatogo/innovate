"""Tests for the preprocess module."""

import numpy as np
import pandas as pd
import pytest

from src.innovate.preprocess.decomposition import stl_decomposition
from src.innovate.preprocess.time_series import rolling_average, sarima_fit
from src.innovate.utils.preprocessing import (
    aggregate_time_series,
    apply_rolling_average,
    apply_sarima,
    apply_stl_decomposition,
    cumulative_sum,
    ensure_datetime_index,
)


class TestSTLDecomposition:
    """Test cases for stl_decomposition function."""

    def test_stl_decomposition_with_datetime_index(self):
        """Test STL decomposition with a proper datetime index."""
        # Create a time series with a datetime index
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        data = pd.Series(np.random.randn(100), index=dates)

        # Add some trend and seasonality
        trend = np.linspace(0, 10, 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
        data = pd.Series(trend + seasonal + np.random.normal(0, 1, 100), index=dates)

        result = stl_decomposition(data, period=7)

        # Check that result is a DataFrame with the expected columns
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["trend", "seasonal", "residual"]
        assert len(result) == len(data)

    def test_stl_decomposition_invalid_index_type(self):
        """Test STL decomposition with non-datetime index raises error."""
        # Create a time series with integer index
        data = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])

        with pytest.raises(TypeError, match="must have a DatetimeIndex"):
            stl_decomposition(data, period=7)


class TestRollingAverage:
    """Test cases for rolling_average function."""

    def test_rolling_average_basic(self):
        """Test rolling average with basic parameters."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3

        result = rolling_average(series, window)

        # First two values should be NaN due to insufficient data for window
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])  # Third value should have a result

        # Check that the rolling average calculation is correct
        expected = series.rolling(window=window).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_rolling_average_window_one(self):
        """Test rolling average with window size of 1."""
        series = pd.Series([5, 10, 15, 20])
        window = 1

        result = rolling_average(series, window)
        expected = series.rolling(window=window).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestSARIMAFit:
    """Test cases for sarima_fit function."""

    def test_sarima_fit_basic(self):
        """Test SARIMA fit with basic parameters."""
        # Create a simple time series
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        np.random.seed(42)  # For reproducible results
        series = pd.Series(np.random.randn(50), index=dates)

        order = (1, 1, 1)  # (p, d, q)
        seasonal_order = (1, 1, 1, 7)  # (P, D, Q, s)

        result = sarima_fit(series, order, seasonal_order)

        # Result should be a pandas Series with same length as input
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        # The fitted values should have been calculated (no NaN values)
        assert not result.isna().all()

    def test_apply_sarima_invalid_order_negative(self):
        """Test SARIMA with negative values in order raises ValueError."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        series = pd.Series(np.random.randn(30), index=dates)

        # Negative order parameters should raise ValueError
        order = (-1, 1, 1)
        seasonal_order = (1, 1, 1, 7)

        with pytest.raises(ValueError):
            apply_sarima(series, order, seasonal_order)

    def test_apply_sarima_invalid_seasonal_order_negative(self):
        """Test SARIMA with negative values in seasonal_order raises ValueError."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        series = pd.Series(np.random.randn(30), index=dates)

        order = (1, 1, 1)
        # Negative seasonal_order parameters should raise ValueError
        seasonal_order = (-1, 1, 1, 7)

        with pytest.raises(ValueError):
            apply_sarima(series, order, seasonal_order)

    def test_apply_sarima_invalid_order_length(self):
        """Test SARIMA with incorrect order tuple length raises ValueError."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        series = pd.Series(np.random.randn(30), index=dates)

        # Incorrect length for order tuple (should be 3)
        order = (1, 1)
        seasonal_order = (1, 1, 1, 7)

        with pytest.raises(ValueError):
            apply_sarima(series, order, seasonal_order)

    def test_apply_sarima_invalid_seasonal_order_length(self):
        """Test SARIMA with incorrect seasonal_order tuple length raises ValueError."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        series = pd.Series(np.random.randn(30), index=dates)

        order = (1, 1, 1)
        # Incorrect length for seasonal_order tuple (should be 4)
        seasonal_order = (1, 1, 1)

        with pytest.raises(ValueError):
            apply_sarima(series, order, seasonal_order)

    def test_apply_sarima_invalid_order_type(self):
        """Test SARIMA with invalid type in order raises ValueError."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        series = pd.Series(np.random.randn(30), index=dates)

        # Invalid type for order parameter (e.g., string)
        order = ("a", 1, 1)
        seasonal_order = (1, 1, 1, 7)

        with pytest.raises(ValueError):
            apply_sarima(series, order, seasonal_order)


class TestEnsureDatetimeIndex:
    """Test cases for ensure_datetime_index function."""

    def test_ensure_datetime_index_with_datetime_index(self):
        """Test function with data that already has a datetime index."""
        dates = pd.date_range(start="2020-01-01", periods=5)
        data = pd.Series([1, 2, 3, 4, 5], index=dates)

        result = ensure_datetime_index(data)
        pd.testing.assert_series_equal(result, data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_ensure_datetime_index_with_string_index(self):
        """Test function converting string index to datetime index."""
        data = pd.Series([1, 2, 3, 4, 5], index=["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])

        result = ensure_datetime_index(data)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == len(data)

    def test_ensure_datetime_index_with_numeric_index(self):
        """Test function with numeric index that represents dates."""
        data = pd.Series([1, 2, 3, 4, 5], index=[2020, 2021, 2022, 2023, 2024])

        result = ensure_datetime_index(data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_ensure_datetime_index_with_invalid_strings(self):
        """Test function with strings that can't be converted to dates."""
        data = pd.Series([1, 2, 3], index=["not_a_date", "also_not", "invalid"])

        with pytest.raises(ValueError, match="Could not convert index to DatetimeIndex"):
            ensure_datetime_index(data)

    def test_ensure_datetime_index_dataframe(self):
        """Test function with DataFrame as input."""
        dates = pd.date_range(start="2020-01-01", periods=3)
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=dates)

        # Already has datetime index
        result = ensure_datetime_index(df)
        pd.testing.assert_frame_equal(result, df)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestAggregateTimeSeries:
    """Test cases for aggregate_time_series function."""

    def test_aggregate_time_series_daily_to_weekly(self):
        """Test aggregation from daily to weekly frequency."""
        dates = pd.date_range(start="2020-01-01", periods=14, freq="D")  # Two weeks
        series = pd.Series([1] * 14, index=dates)

        result = aggregate_time_series(series, "W")  # Weekly aggregation

        # Should have 2 aggregated weeks
        assert len(result) == 2
        assert result.iloc[0] == 7  # First week sum
        assert result.iloc[1] == 7  # Second week sum

    def test_aggregate_time_series_with_datetime_conversion(self):
        """Test aggregation after datetime index conversion."""
        series = pd.Series([1, 2, 3, 4], index=["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-09"])  # 8 days

        result = aggregate_time_series(series, "W")  # Weekly aggregation

        # Should aggregate the first week and second week
        assert len(result) >= 1  # At least one week
        assert result.iloc[0] >= 3  # First week sum (at least first 3 values)


class TestApplySTLDecomposition:
    """Test cases for apply_stl_decomposition function."""

    def test_apply_stl_decomposition_basic(self):
        """Test STL decomposition with basic parameters."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        # Create data with trend and seasonality
        trend = np.linspace(0, 10, 50)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(50) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 0.5, 50)
        data = pd.Series(trend + seasonal + noise, index=dates)

        trend_comp, seasonal_comp, resid_comp = apply_stl_decomposition(data, period=7)

        # All components should be pandas Series with same length
        assert isinstance(trend_comp, pd.Series)
        assert isinstance(seasonal_comp, pd.Series)
        assert isinstance(resid_comp, pd.Series)
        assert len(trend_comp) == len(data)
        assert len(seasonal_comp) == len(data)
        assert len(resid_comp) == len(data)

        # The original data should roughly equal the sum of components
        reconstructed = trend_comp + seasonal_comp + resid_comp
        pd.testing.assert_series_equal(reconstructed, data, check_names=False)

    def test_apply_stl_decomposition_infer_period(self):
        """Test STL decomposition with period inference."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        data = pd.Series(np.random.randn(30) + np.linspace(0, 5, 30), index=dates)

        # With 30 days of data, it should infer a period (default to 12 for monthly)
        trend_comp, seasonal_comp, resid_comp = apply_stl_decomposition(data, period=None)

        assert isinstance(trend_comp, pd.Series)
        assert isinstance(seasonal_comp, pd.Series)
        assert isinstance(resid_comp, pd.Series)

    def test_apply_stl_decomposition_short_data_no_period(self):
        """Test STL decomposition with short data and no period raises error."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        data = pd.Series(np.random.randn(5), index=dates)

        with pytest.raises(ValueError, match="Period must be specified"):
            apply_stl_decomposition(data, period=None)


class TestCumulativeSum:
    """Test cases for cumulative_sum function."""

    def test_cumulative_sum_basic(self):
        """Test cumulative sum with basic input."""
        data = [1, 2, 3, 4, 5]
        result = cumulative_sum(data)

        expected = np.array([1, 3, 6, 10, 15])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_sum_empty_list(self):
        """Test cumulative sum with empty list."""
        data = []
        result = cumulative_sum(data)

        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_sum_single_element(self):
        """Test cumulative sum with single element."""
        data = [5]
        result = cumulative_sum(data)

        expected = np.array([5])
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_sum_with_numpy_array(self):
        """Test cumulative sum with numpy array input."""
        data = np.array([1, 2, 3])
        result = cumulative_sum(data)

        expected = np.array([1, 3, 6])
        np.testing.assert_array_equal(result, expected)


class TestApplyRollingAverage:
    """Test cases for apply_rolling_average function."""

    def test_apply_rolling_average_basic(self):
        """Test rolling average with basic parameters."""
        series = pd.Series([1, 2, 3, 4, 5, 6])
        window = 3

        result = apply_rolling_average(series, window)
        expected = series.rolling(window=window).mean()

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_apply_rolling_average_window_larger_than_data(self):
        """Test rolling average with window larger than data."""
        series = pd.Series([1, 2, 3])
        window = 5  # Larger than data length

        result = apply_rolling_average(series, window)

        # All values should be NaN since window is larger than data
        assert result.isna().all()


class TestApplySARIMA:
    """Test cases for apply_sarima function."""

    def test_apply_sarima_basic(self):
        """Test SARIMA with basic parameters."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        np.random.seed(42)
        series = pd.Series(np.random.randn(30), index=dates)

        order = (1, 1, 1)  # (p, d, q)
        seasonal_order = (1, 1, 1, 7)  # (P, D, Q, s)

        result = apply_sarima(series, order, seasonal_order)

        # Result should be a pandas Series with same length as input
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
