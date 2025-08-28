"""Test coverage for backend functionality and utility functions."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from innovate import backend
from innovate.backend import use_backend, current_backend
from innovate.backends.numpy_backend import NumPyBackend
from innovate.utils import metrics, preprocessing, model_evaluation
from innovate.utils.metrics import *
from innovate.utils.preprocessing import *
from innovate.utils.model_evaluation import *


class TestBackendFunctionality:
    """Comprehensive tests for backend switching and functionality."""
    
    def setup_method(self):
        """Reset backend to numpy before each test."""
        use_backend("numpy")
    
    def test_numpy_backend_operations(self):
        """Test NumPy backend operations."""
        backend_np = NumPyBackend()
        
        # Test basic operations
        arr = backend_np.array([1, 2, 3, 4, 5])
        assert isinstance(arr, np.ndarray)
        
        # Test zeros and ones
        zeros = backend_np.zeros(5)
        assert np.all(zeros == 0)
        assert len(zeros) == 5
        
        ones = backend_np.ones(3)
        assert np.all(ones == 1)
        assert len(ones) == 3
        
        # Test zeros_like and ones_like
        zeros_like = backend_np.zeros_like(arr)
        assert zeros_like.shape == arr.shape
        assert np.all(zeros_like == 0)
        
        ones_like = backend_np.ones_like(arr) 
        assert ones_like.shape == arr.shape
        assert np.all(ones_like == 1)
        
        # Test mathematical operations
        assert backend_np.sum(arr) == 15
        assert backend_np.mean(arr) == 3.0
        
        # Test exponential and logarithmic
        exp_arr = backend_np.exp(np.array([0, 1, 2]))
        assert np.allclose(exp_arr, np.array([1, np.e, np.e**2]))
        
        # Test where function
        condition = arr > 3
        result = backend_np.where(condition, arr, 0)
        expected = np.array([0, 0, 0, 4, 5])
        np.testing.assert_array_equal(result, expected)
        
        # Test minimum and maximum
        assert backend_np.min(arr) == 1
        assert backend_np.max(arr) == 5
        
        # Test matrix multiplication
        mat1 = backend_np.array([[1, 2], [3, 4]])
        mat2 = backend_np.array([[5, 6], [7, 8]])
        result = backend_np.matmul(mat1, mat2)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_backend_switching(self):
        """Test switching between backends."""
        # Start with numpy
        use_backend("numpy")
        assert isinstance(current_backend, NumPyBackend)
        
        # Test invalid backend
        with pytest.raises(ValueError, match="Unknown backend"):
            use_backend("invalid")
        
        # Test JAX backend if available
        try:
            use_backend("jax")
            # If successful, test basic operations
            arr = current_backend.array([1, 2, 3])
            assert len(arr) == 3
            
            # Switch back to numpy
            use_backend("numpy")
            assert isinstance(current_backend, NumPyBackend)
            
        except ImportError:
            # JAX not available, which is fine
            pass

    def test_backend_error_handling(self):
        """Test backend error handling."""
        # Test with None backend name
        with pytest.raises((ValueError, TypeError)):
            use_backend(None)
        
        # Test with empty string
        with pytest.raises(ValueError):
            use_backend("")

    @patch('innovate.backends.jax_backend.JaxBackend', None)
    def test_jax_backend_unavailable(self):
        """Test behavior when JAX backend is not available."""
        # Temporarily set JaxBackend to None
        original_jax = backend.JaxBackend
        backend.JaxBackend = None
        
        try:
            with pytest.raises(ImportError, match="JAX backend is not available"):
                use_backend("jax")
        finally:
            # Restore original value
            backend.JaxBackend = original_jax


class TestMetricsComprehensive:
    """Comprehensive tests for metrics calculations."""
    
    def test_mse_calculation(self):
        """Test MSE calculation with various inputs."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        mse = calculate_mse(y_true, y_pred)
        expected = np.mean([(1-1.1)**2, (2-1.9)**2, (3-3.1)**2, (4-3.9)**2, (5-5.1)**2])
        assert np.isclose(mse, expected)
        
        # Perfect predictions
        mse_perfect = calculate_mse(y_true, y_true)
        assert mse_perfect == 0.0
        
        # Test with numpy arrays
        mse_np = calculate_mse(np.array(y_true), np.array(y_pred))
        assert np.isclose(mse, mse_np)

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = [0, 1, 4, 9, 16]
        y_pred = [0.1, 0.9, 4.1, 8.9, 16.1]
        
        rmse = calculate_rmse(y_true, y_pred)
        mse = calculate_mse(y_true, y_pred)
        expected = np.sqrt(mse)
        assert np.isclose(rmse, expected)

    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.5, 1.5, 3.5, 3.5, 5.5]
        
        mae = calculate_mae(y_true, y_pred)
        expected = np.mean([0.5, 0.5, 0.5, 0.5, 0.5])
        assert np.isclose(mae, expected)

    def test_r_squared_calculation(self):
        """Test R-squared calculation with edge cases."""
        # Perfect correlation
        y_true = [1, 2, 3, 4, 5]
        r2_perfect = calculate_r_squared(y_true, y_true)
        assert np.isclose(r2_perfect, 1.0)
        
        # No correlation (predict mean)
        y_mean = [np.mean(y_true)] * len(y_true)
        r2_mean = calculate_r_squared(y_true, y_mean)
        assert np.isclose(r2_mean, 0.0, atol=1e-10)
        
        # Worse than mean prediction (negative R²)
        y_bad = [10] * len(y_true)  # Constant far from data
        r2_bad = calculate_r_squared(y_true, y_bad)
        assert r2_bad < 0
        
        # Constant true values (edge case)
        y_constant = [5] * 5
        r2_constant = calculate_r_squared(y_constant, y_constant)
        assert r2_constant == 0.0  # SS_tot = 0, function returns 0

    def test_mape_calculation(self):
        """Test MAPE calculation with edge cases."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 1.8, 3.3, 3.6, 5.5]
        
        mape = calculate_mape(y_true, y_pred)
        expected = np.mean([10, 10, 10, 10, 10])  # Each 10% error
        assert np.isclose(mape, expected)
        
        # Test with zeros (should handle division by zero)
        y_true_with_zero = [0, 1, 2, 3]
        y_pred_with_zero = [0.1, 1.1, 1.9, 3.1]
        
        # MAPE with zeros should handle gracefully
        mape_with_zero = calculate_mape(y_true_with_zero, y_pred_with_zero)
        assert np.isfinite(mape_with_zero)

    def test_smape_calculation(self):
        """Test SMAPE calculation."""
        y_true = [2, 4, 6, 8]
        y_pred = [3, 3, 7, 7]
        
        smape = calculate_smape(y_true, y_pred)
        
        # Manual calculation
        expected = np.mean([
            abs(3-2) / ((abs(2) + abs(3))/2) * 100,
            abs(3-4) / ((abs(4) + abs(3))/2) * 100,
            abs(7-6) / ((abs(6) + abs(7))/2) * 100,
            abs(7-8) / ((abs(8) + abs(7))/2) * 100
        ])
        assert np.isclose(smape, expected)
        
        # Test with zeros
        y_zero = [0, 0]
        y_pred_zero = [0, 0]
        smape_zero = calculate_smape(y_zero, y_pred_zero)
        assert smape_zero == 0.0

    def test_rss_calculation(self):
        """Test RSS calculation."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        rss = calculate_rss(y_true, y_pred)
        expected = sum([(1-1.1)**2, (2-1.9)**2, (3-3.1)**2, (4-3.9)**2, (5-5.1)**2])
        assert np.isclose(rss, expected)

    def test_aic_bic_calculation(self):
        """Test AIC and BIC calculations."""
        n_params = 3
        n_samples = 100
        rss = 50.0
        
        aic = calculate_aic(n_params, n_samples, rss)
        expected_aic = 2 * n_params + n_samples * np.log(rss / n_samples)
        assert np.isclose(aic, expected_aic)
        
        bic = calculate_bic(n_params, n_samples, rss)
        expected_bic = np.log(n_samples) * n_params + n_samples * np.log(rss / n_samples)
        assert np.isclose(bic, expected_bic)

    def test_metrics_with_empty_inputs(self):
        """Test metrics with empty inputs."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_mse([], [])
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_r_squared([], [])

    def test_metrics_with_mismatched_lengths(self):
        """Test metrics with mismatched input lengths."""
        y_true = [1, 2, 3]
        y_pred = [1, 2]  # Different length
        
        with pytest.raises((ValueError, IndexError)):
            calculate_mse(y_true, y_pred)
        
        with pytest.raises((ValueError, IndexError)):
            calculate_r_squared(y_true, y_pred)


class TestPreprocessingFunctions:
    """Test preprocessing utility functions."""
    
    def test_ensure_datetime_index_success(self):
        """Test successful datetime index conversion."""
        # Numeric index that can be converted
        dates_numeric = pd.Series([1, 2, 3], index=[20200101, 20200102, 20200103])
        result = ensure_datetime_index(dates_numeric)
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # String index that can be converted
        dates_string = pd.Series([1, 2, 3], index=['2020-01-01', '2020-01-02', '2020-01-03'])
        result = ensure_datetime_index(dates_string)
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # Already datetime index
        dates_dt = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        result = ensure_datetime_index(dates_dt)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_ensure_datetime_index_failure(self):
        """Test datetime index conversion failure."""
        # Non-convertible index
        invalid_series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        
        with pytest.raises(ValueError, match="Could not convert index to DatetimeIndex"):
            ensure_datetime_index(invalid_series)

    def test_aggregate_time_series(self):
        """Test time series aggregation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.Series(range(10), index=dates)
        
        # Aggregate to weekly
        weekly = aggregate_time_series(data, 'W')
        assert len(weekly) <= len(data)
        
        # Aggregate to monthly
        monthly = aggregate_time_series(data, 'M')
        assert len(monthly) <= len(data)

    def test_apply_stl_decomposition_success(self):
        """Test successful STL decomposition."""
        # Create seasonal data
        dates = pd.date_range('2020-01-01', periods=48, freq='M')
        trend = np.linspace(100, 200, 48)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(48) / 12)
        data = pd.Series(trend + seasonal, index=dates)
        
        trend_result, seasonal_result, resid_result = apply_stl_decomposition(data, period=12)
        
        assert len(trend_result) == len(data)
        assert len(seasonal_result) == len(data)  
        assert len(resid_result) == len(data)
        
        # Reconstruction should approximately equal original
        reconstructed = trend_result + seasonal_result + resid_result
        np.testing.assert_allclose(reconstructed.values, data.values, rtol=1e-10)

    def test_apply_stl_decomposition_auto_period(self):
        """Test STL decomposition with automatic period detection."""
        # Long enough series for auto-detection
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        data = pd.Series(np.random.randn(50), index=dates)
        
        # Should default to period=12 for long series
        trend, seasonal, resid = apply_stl_decomposition(data, period=None)
        assert len(trend) == len(data)

    def test_apply_stl_decomposition_too_short(self):
        """Test STL decomposition with too short series."""
        dates = pd.date_range('2020-01-01', periods=5, freq='M')
        data = pd.Series(np.random.randn(5), index=dates)
        
        with pytest.raises(ValueError, match="Period must be specified"):
            apply_stl_decomposition(data, period=None)

    def test_apply_rolling_average(self):
        """Test rolling average calculation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.Series(range(10), index=dates)
        
        # 3-point rolling average
        smoothed = apply_rolling_average(data, window=3)
        assert len(smoothed) == len(data)
        assert smoothed.isnull().sum() == 2  # First 2 values should be NaN
        
        # Check specific values
        assert smoothed.iloc[2] == 1.0  # (0+1+2)/3
        assert smoothed.iloc[3] == 2.0  # (1+2+3)/3

    def test_apply_sarima(self):
        """Test SARIMA fitting."""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        # Create data with trend and seasonality
        trend = np.linspace(100, 150, 50)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(50) / 12)
        noise = np.random.normal(0, 2, 50)
        data = pd.Series(trend + seasonal + noise, index=dates)
        
        # Simple ARIMA order
        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0)
        
        fitted_values = apply_sarima(data, order=order, seasonal_order=seasonal_order)
        assert len(fitted_values) == len(data)
        assert np.all(np.isfinite(fitted_values))

    def test_cumulative_sum(self):
        """Test cumulative sum calculation."""
        data = [1, 2, 3, 4, 5]
        cumsum = cumulative_sum(data)
        expected = np.array([1, 3, 6, 10, 15])
        np.testing.assert_array_equal(cumsum, expected)
        
        # Empty input
        cumsum_empty = cumulative_sum([])
        assert len(cumsum_empty) == 0

    def test_preprocessing_with_dataframes(self):
        """Test preprocessing functions with DataFrames."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20)
        }, index=dates)
        
        # Test ensure_datetime_index with DataFrame
        result_df = ensure_datetime_index(df)
        assert isinstance(result_df.index, pd.DatetimeIndex)
        
        # Test aggregation with DataFrame
        weekly_df = aggregate_time_series(df, 'W')
        assert isinstance(weekly_df, pd.DataFrame)
        assert len(weekly_df) <= len(df)


class TestModelEvaluationUtilities:
    """Test model evaluation utility functions."""
    
    def test_compute_residuals(self):
        """Test residual computation."""
        from innovate.diffuse.bass import BassModel
        
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        t = [1, 2, 3, 4, 5]
        y = [10, 25, 45, 70, 100]
        
        residuals = compute_residuals(model, t, y)
        predictions = model.predict(t)
        expected_residuals = np.array(y) - np.array(predictions)
        
        np.testing.assert_array_almost_equal(residuals, expected_residuals)

    def test_compute_residuals_unfitted(self):
        """Test residuals computation with unfitted model."""
        from innovate.diffuse.bass import BassModel
        
        model = BassModel()  # Not fitted
        t = [1, 2, 3]
        y = [10, 20, 30]
        
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            compute_residuals(model, t, y)

    def test_residual_acf_pacf(self):
        """Test residual ACF and PACF calculation."""
        from innovate.diffuse.logistic import LogisticModel
        from innovate.fitters.scipy_fitter import ScipyFitter
        
        # Create and fit model
        t = np.linspace(0, 20, 50)
        y_true = 100 / (1 + np.exp(-0.5 * (t - 10)))
        y_noisy = y_true + np.random.normal(0, 2, len(t))
        
        model = LogisticModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y_noisy)
        
        # Test ACF
        acf_values = residual_acf(model, t, y_noisy, nlags=5)
        assert len(acf_values) == 6  # nlags + 1 (lag 0)
        assert acf_values[0] == 1.0  # ACF at lag 0 should be 1
        
        # Test PACF
        pacf_values = residual_pacf(model, t, y_noisy, nlags=5)
        assert len(pacf_values) == 6

    def test_compare_models_functionality(self):
        """Test model comparison functionality."""
        from innovate.diffuse.bass import BassModel
        from innovate.diffuse.logistic import LogisticModel
        from innovate.fitters.scipy_fitter import ScipyFitter
        
        # Generate data and fit multiple models
        t = np.linspace(1, 20, 30)
        y_true = 100 / (1 + np.exp(-0.3 * (t - 10)))
        y_noisy = y_true + np.random.normal(0, 3, len(t))
        
        # Fit Bass model
        bass = BassModel()
        fitter = ScipyFitter()
        fitter.fit(bass, t, y_noisy)
        
        # Fit Logistic model  
        logistic = LogisticModel()
        fitter.fit(logistic, t, y_noisy)
        
        models = [bass, logistic]
        model_names = ["Bass", "Logistic"]
        
        # Compare models
        comparison_df = compare_models(models, model_names, t, y_noisy)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert "Model" in comparison_df.index.names or "Model" in comparison_df.columns
        assert "RMSE" in comparison_df.columns
        assert "R_squared" in comparison_df.columns

    def test_find_best_model(self):
        """Test finding best model from comparison."""
        # Mock comparison DataFrame
        comparison_df = pd.DataFrame({
            'RMSE': [10.0, 15.0, 8.0],
            'R_squared': [0.8, 0.7, 0.85],
            'AIC': [100, 120, 95]
        }, index=['Model_A', 'Model_B', 'Model_C'])
        
        # Best by RMSE (minimize)
        best_name, best_row = find_best_model(comparison_df, metric='RMSE', minimize=True)
        assert best_name == 'Model_C'
        assert best_row['RMSE'] == 8.0
        
        # Best by R² (maximize)
        best_name, best_row = find_best_model(comparison_df, metric='R_squared', minimize=False)
        assert best_name == 'Model_C'
        assert best_row['R_squared'] == 0.85

    def test_find_best_model_invalid_metric(self):
        """Test find_best_model with invalid metric."""
        comparison_df = pd.DataFrame({'RMSE': [10, 15, 8]}, index=['A', 'B', 'C'])
        
        with pytest.raises(ValueError, match="Metric 'InvalidMetric' not found"):
            find_best_model(comparison_df, metric='InvalidMetric')

    def test_model_evaluation_edge_cases(self):
        """Test model evaluation with edge cases."""
        from innovate.diffuse.bass import BassModel
        
        # Model with extreme parameters
        model = BassModel()
        model.params_ = {"p": 1e-10, "q": 1e-10, "m": 1e10}
        
        t = [1, 2, 3]
        y = [100, 200, 300]
        
        # Should handle extreme parameters gracefully
        metrics = get_fit_metrics(model, t, y)
        assert isinstance(metrics, dict)
        assert all(np.isfinite(v) for v in metrics.values() if v is not None)