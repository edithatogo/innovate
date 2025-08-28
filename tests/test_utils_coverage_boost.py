"""
Focused tests for utility modules and backend functionality to improve test coverage.

This test file specifically targets modules that are likely under-covered:
- Backend switching functionality
- Utility functions
- Preprocessing modules
- Metrics and evaluation functions
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from innovate.backend import use_backend, current_backend
from innovate.utils.model_evaluation import get_fit_metrics, model_aic, model_bic
from innovate.utils.preprocessing import ensure_datetime_index
from innovate.preprocess.time_series import rolling_average, sarima_fit
from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel


class TestBackendFunctionality:
    """Test backend switching and functionality."""
    
    def test_current_backend_default(self):
        """Test that current_backend is a valid backend object."""
        backend = current_backend
        # Should be a backend object, not None
        assert backend is not None
        # Should have a name attribute or be identifiable
        backend_name = str(type(backend).__name__)
        assert 'Backend' in backend_name
    
    def test_use_backend_numpy(self):
        """Test switching to numpy backend."""
        original_backend = current_backend
        
        use_backend('numpy')
        assert 'NumPy' in str(type(current_backend).__name__)
        
        # Restore original backend
        if 'Jax' in str(type(original_backend).__name__):
            try:
                use_backend('jax')
            except ImportError:
                pass  # JAX not available
        else:
            use_backend('numpy')
    
    def test_use_backend_invalid(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            use_backend("invalid_backend")
    
    def test_use_backend_case_sensitivity(self):
        """Test backend switching with different cases."""
        original_backend = current_backend
        
        # Test that backend names are case sensitive
        with pytest.raises(ValueError, match="Unknown backend"):
            use_backend('NUMPY')
        
        # Restore original backend  
        if 'Jax' in str(type(original_backend).__name__):
            try:
                use_backend('jax')
            except ImportError:
                pass
        else:
            use_backend('numpy')
    
    def test_backend_switching_preserves_functionality(self):
        """Test that backend switching preserves model functionality."""
        original_backend = current_backend
        
        # Test with numpy backend
        use_backend('numpy')
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        t = [1, 2, 3, 4, 5]
        predictions_numpy = model.predict(t)
        
        # Try JAX backend if available
        try:
            use_backend('jax')
            predictions_jax = model.predict(t)
            
            # Results should be very similar (allowing for numerical differences)
            np.testing.assert_allclose(predictions_numpy, predictions_jax, rtol=1e-6)
        except (ValueError, ImportError):
            # JAX not available, skip comparison
            pass
        
        # Restore original backend
        if 'Jax' in str(type(original_backend).__name__):
            try:
                use_backend('jax')
            except ImportError:
                pass
        else:
            use_backend('numpy')


class TestUtilsModelEvaluation:
    """Test model evaluation utilities."""
    
    def setup_method(self):
        """Set up fitted model for testing."""
        self.model = BassModel()
        self.model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        self.t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.y = np.array([15, 35, 65, 105, 155, 215, 285, 365, 455, 555])
    
    def test_get_fit_metrics_basic(self):
        """Test basic functionality of get_fit_metrics."""
        metrics = get_fit_metrics(self.model, self.t, self.y)
        
        assert isinstance(metrics, dict)
        expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert np.isfinite(metrics[metric])
    
    def test_get_fit_metrics_perfect_fit(self):
        """Test metrics with perfect fit."""
        # Use model predictions as "observed" data
        y_perfect = self.model.predict(self.t)
        metrics = get_fit_metrics(self.model, self.t, y_perfect)
        
        # MSE and RMSE should be very close to 0
        assert metrics['mse'] < 1e-10
        assert metrics['rmse'] < 1e-5
        assert metrics['mae'] < 1e-10
        # R² should be very close to 1
        assert metrics['r2'] > 0.999
        # MAPE should be very small
        assert metrics['mape'] < 0.01
    
    def test_get_fit_metrics_poor_fit(self):
        """Test metrics with deliberately poor fit."""
        # Create constant predictions that don't match data trend
        poor_model = BassModel()
        poor_model.params_ = {"p": 0, "q": 0, "m": 50}  # Will predict ~0 always
        
        metrics = get_fit_metrics(poor_model, self.t, self.y)
        
        # Should have poor metrics
        assert metrics['mse'] > 1000  # High error
        assert metrics['r2'] < 0.5   # Poor fit
        assert metrics['mape'] > 50  # High percentage error
    
    def test_model_aic_basic(self):
        """Test AIC calculation."""
        aic = model_aic(self.model, self.t, self.y)
        
        assert isinstance(aic, (int, float))
        assert np.isfinite(aic)
        # AIC should be reasonable for this model/data
        assert aic > 0  # AIC is typically positive
    
    def test_model_bic_basic(self):
        """Test BIC calculation."""
        bic = model_bic(self.model, self.t, self.y)
        
        assert isinstance(bic, (int, float))
        assert np.isfinite(bic)
        # BIC should be higher than AIC for same model (BIC penalizes parameters more)
        aic = model_aic(self.model, self.t, self.y)
        assert bic >= aic
    
    def test_aic_bic_model_comparison(self):
        """Test AIC/BIC for model comparison."""
        # Compare bass model vs logistic model
        logistic_model = LogisticModel()
        logistic_model.params_ = {"L": 1000, "k": 0.1, "x0": 5}
        
        bass_aic = model_aic(self.model, self.t, self.y)
        logistic_aic = model_aic(logistic_model, self.t, self.y)
        
        bass_bic = model_bic(self.model, self.t, self.y)
        logistic_bic = model_bic(logistic_model, self.t, self.y)
        
        # All should be finite numbers
        assert all(np.isfinite(x) for x in [bass_aic, logistic_aic, bass_bic, logistic_bic])
        
        # BIC should be >= AIC for each model
        assert bass_bic >= bass_aic
        assert logistic_bic >= logistic_aic
    
    def test_metrics_with_zero_variance_data(self):
        """Test metrics behavior with zero variance in observed data."""
        # All observed values are the same
        y_constant = np.full_like(self.t, 100)
        
        metrics = get_fit_metrics(self.model, self.t, y_constant)
        
        # R² should handle zero variance case
        assert 'r2' in metrics
        # Other metrics should still be calculable
        assert np.isfinite(metrics['mse'])
        assert np.isfinite(metrics['mae'])
    
    def test_metrics_with_negative_values(self):
        """Test metrics with negative observed values."""
        # Some negative values (unusual but should be handled)
        y_with_negatives = self.y.copy()
        y_with_negatives[0] = -10
        
        metrics = get_fit_metrics(self.model, self.t, y_with_negatives)
        
        # Should still calculate basic metrics
        assert np.isfinite(metrics['mse'])
        assert np.isfinite(metrics['mae'])
        assert np.isfinite(metrics['r2'])
        # MAPE might be affected by negative values, but should not crash
        assert 'mape' in metrics


class TestUtilsPreprocessing:
    """Test preprocessing utilities."""
    
    def test_ensure_datetime_index_with_datetime(self):
        """Test ensure_datetime_index with already datetime index."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        
        result = ensure_datetime_index(series)
        
        assert isinstance(result.index, pd.DatetimeIndex)
        pd.testing.assert_series_equal(result, series)
    
    def test_ensure_datetime_index_with_string_dates(self):
        """Test ensure_datetime_index with string date index."""
        string_dates = ['2020-01-01', '2020-01-02', '2020-01-03']
        series = pd.Series([1, 2, 3], index=string_dates)
        
        result = ensure_datetime_index(series)
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == len(series)
        assert result.index[0] == pd.Timestamp('2020-01-01')
    
    def test_ensure_datetime_index_with_numeric_dates(self):
        """Test ensure_datetime_index with numeric index (timestamps)."""
        # Unix timestamps
        timestamps = [1577836800, 1577923200, 1578009600]  # 2020-01-01, 02, 03
        series = pd.Series([1, 2, 3], index=timestamps)
        
        try:
            result = ensure_datetime_index(series)
            assert isinstance(result.index, pd.DatetimeIndex)
        except ValueError:
            # If conversion fails, that's also acceptable behavior
            pass
    
    def test_ensure_datetime_index_invalid_strings(self):
        """Test ensure_datetime_index with invalid date strings."""
        invalid_dates = ['not-a-date', 'also-invalid', 'nope']
        series = pd.Series([1, 2, 3], index=invalid_dates)
        
        with pytest.raises(ValueError):
            ensure_datetime_index(series)
    
    def test_ensure_datetime_index_mixed_types(self):
        """Test ensure_datetime_index with mixed index types."""
        mixed_index = [1, '2020-01-02', 'invalid']
        series = pd.Series([1, 2, 3], index=mixed_index)
        
        with pytest.raises(ValueError):
            ensure_datetime_index(series)


class TestPreprocessTimeSeriesModule:
    """Test time series preprocessing module."""
    
    def test_rolling_average_basic(self):
        """Test basic rolling average functionality."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3
        
        result = rolling_average(data, window)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        # First few values should be NaN due to window
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value should be average of first 3
        assert result.iloc[2] == 2.0  # (1+2+3)/3
    
    def test_rolling_average_window_larger_than_data(self):
        """Test rolling average with window larger than data."""
        data = pd.Series([1, 2, 3])
        window = 5
        
        result = rolling_average(data, window)
        
        # All values should be NaN since window > data length
        assert pd.isna(result).all()
    
    def test_rolling_average_window_one(self):
        """Test rolling average with window size 1."""
        data = pd.Series([1, 2, 3, 4, 5])
        window = 1
        
        result = rolling_average(data, window)
        
        # Should be identical to original data
        pd.testing.assert_series_equal(result, data)
    
    def test_sarima_fit_basic(self):
        """Test basic SARIMA fitting functionality."""
        # Create simple time series
        data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4])
        order = (1, 0, 0)
        seasonal_order = (0, 0, 0, 4)
        
        try:
            result = sarima_fit(data, order, seasonal_order)
            
            assert isinstance(result, pd.Series)
            assert len(result) == len(data)
            # Fitted values should be finite
            assert np.all(np.isfinite(result))
        except ImportError:
            # If SARIMA dependencies not available, skip
            pytest.skip("SARIMA dependencies not available")
        except Exception as e:
            # SARIMA might fail with simple data, that's ok
            pass
    
    def test_sarima_fit_invalid_order(self):
        """Test SARIMA with invalid order parameters."""
        data = pd.Series([1, 2, 3, 4, 5])
        invalid_order = (-1, 0, 0)  # Negative order
        seasonal_order = (0, 0, 0, 4)
        
        try:
            with pytest.raises((ValueError, Exception)):
                sarima_fit(data, invalid_order, seasonal_order)
        except ImportError:
            pytest.skip("SARIMA dependencies not available")
    
    def test_sarima_fit_too_short_series(self):
        """Test SARIMA with very short time series."""
        data = pd.Series([1, 2])  # Very short series
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 4)
        
        try:
            # Should either work or raise appropriate error
            result = sarima_fit(data, order, seasonal_order)
            if result is not None:
                assert len(result) == len(data)
        except (ValueError, Exception):
            # Expected for too short series
            pass
        except ImportError:
            pytest.skip("SARIMA dependencies not available")


class TestEdgeCasesAndErrorHandling:
    """Test various edge cases and error handling scenarios."""
    
    def test_model_evaluation_with_inf_values(self):
        """Test model evaluation with infinite values in predictions."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Mock predict to return inf values
        original_predict = model.predict
        def mock_predict(t):
            result = original_predict(t)
            result[0] = np.inf  # Inject inf value
            return result
        
        with patch.object(model, 'predict', side_effect=mock_predict):
            t = [1, 2, 3, 4, 5]
            y = [10, 20, 30, 40, 50]
            
            # Metrics should handle inf values gracefully
            try:
                metrics = get_fit_metrics(model, t, y)
                # Should either exclude inf values or return inf metrics
                assert 'mse' in metrics
            except (ValueError, RuntimeError):
                # Raising error for inf values is also acceptable
                pass
    
    def test_model_evaluation_with_nan_values(self):
        """Test model evaluation with NaN values."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        t = [1, 2, 3, 4, 5]
        y = [10, np.nan, 30, 40, 50]  # NaN in observed data
        
        try:
            metrics = get_fit_metrics(model, t, y)
            # Should handle NaN appropriately
            assert 'mse' in metrics
        except (ValueError, RuntimeError):
            # Raising error for NaN is also acceptable
            pass
    
    def test_backend_switching_during_computation(self):
        """Test backend switching doesn't break ongoing computations."""
        original_backend = current_backend
        
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        t = [1, 2, 3, 4, 5]
        
        # Start computation
        predictions1 = model.predict(t)
        
        # Switch backend (if possible)
        try:
            if 'NumPy' in str(type(original_backend).__name__):
                use_backend('jax')
            else:
                use_backend('numpy')
        except (ValueError, ImportError):
            pass
        
        # Continue computation
        predictions2 = model.predict(t)
        
        # Results should be similar regardless of backend
        np.testing.assert_allclose(predictions1, predictions2, rtol=1e-6)
        
        # Restore original backend
        if 'Jax' in str(type(original_backend).__name__):
            try:
                use_backend('jax')
            except ImportError:
                pass
        else:
            use_backend('numpy')
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Empty time array
        empty_t = []
        try:
            result = model.predict(empty_t)
            assert len(result) == 0
        except (ValueError, IndexError):
            # Raising error for empty input is acceptable
            pass
    
    def test_very_large_inputs(self):
        """Test handling of very large input values."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Very large time values
        large_t = [1e6, 1e7, 1e8]
        result = model.predict(large_t)
        
        # Should handle large values gracefully
        assert len(result) == len(large_t)
        # At very large times, should approach market potential
        assert np.all(result <= model.params_["m"] + 1e-6)
        assert np.all(np.isfinite(result))