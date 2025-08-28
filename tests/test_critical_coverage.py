"""Focused tests for critical edge cases and error handling to improve test coverage."""

import numpy as np
import pandas as pd
import pytest

from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.utils.model_evaluation import get_fit_metrics, model_aic, model_bic
from innovate.utils.preprocessing import ensure_datetime_index
from innovate.policy.intervention import PolicyIntervention
from innovate.backend import use_backend


class TestCriticalEdgeCases:
    """Critical edge case tests for improved coverage."""
    
    def test_models_without_fitted_parameters(self):
        """Test that unfitted models raise appropriate errors."""
        models = [BassModel(), LogisticModel(), GompertzModel()]
        t = [1, 2, 3, 4, 5]
        
        for model in models:
            # Test predict raises error
            with pytest.raises(RuntimeError, match="has not been fitted|parameters are not set"):
                model.predict(t)
            
            # Test score raises error  
            y = [10, 20, 30, 40, 50]
            with pytest.raises(RuntimeError, match="has not been fitted"):
                model.score(t, y)
    
    def test_model_evaluation_unfitted_models(self):
        """Test model evaluation functions with unfitted models."""
        model = BassModel()
        t = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        
        # All evaluation functions should raise RuntimeError
        with pytest.raises(RuntimeError, match="has not been fitted"):
            get_fit_metrics(model, t, y)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model_aic(model, t, y)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model_bic(model, t, y)
    
    def test_extreme_parameter_values(self):
        """Test models with extreme parameter values."""
        model = BassModel()
        
        # Very small parameters (near zero)
        model.params_ = {"p": 1e-10, "q": 1e-10, "m": 1}
        t = [1, 2, 3, 4, 5]
        result = model.predict(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        
        # Zero market potential
        model.params_ = {"p": 0.02, "q": 0.3, "m": 0}
        result = model.predict(t)
        assert np.all(result == 0)
        
        # Zero innovation and imitation (no diffusion)
        model.params_ = {"p": 0, "q": 0, "m": 1000}
        result = model.predict(t)
        assert np.all(result == 0)
    
    def test_invalid_time_inputs(self):
        """Test models with invalid time inputs."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Empty time array
        try:
            result = model.predict([])
            assert len(result) == 0
        except (ValueError, IndexError):
            # Either empty result or error is acceptable
            pass
        
        # Negative times
        t_negative = [-2, -1, 0, 1, 2]
        result = model.predict(t_negative)
        assert len(result) == len(t_negative)
        assert np.all(np.isfinite(result))
    
    def test_preprocessing_edge_cases(self):
        """Test preprocessing functions with edge cases."""
        # Test datetime conversion failure
        invalid_series = pd.Series([1, 2, 3], index=['invalid', 'datetime', 'strings'])
        
        with pytest.raises(ValueError, match="Could not convert index to DatetimeIndex"):
            ensure_datetime_index(invalid_series)
        
        # Test with already valid datetime index
        valid_series = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        result = ensure_datetime_index(valid_series)
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_policy_intervention_errors(self):
        """Test policy intervention error conditions."""
        # Test with unfitted model
        unfitted_model = BassModel()
        policy = PolicyIntervention(unfitted_model)
        
        with pytest.raises(RuntimeError, match="parameters set before applying policy"):
            policy.apply_time_varying_params(t_points=[1, 2, 3])
        
        # Test with unsupported model type
        from innovate.compete.competition import MultiProductDiffusionModel
        multiproduct = MultiProductDiffusionModel(p=[0.02], Q=[[0.1]], m=[1000])
        
        with pytest.raises(TypeError, match="currently only supported for BassModel"):
            PolicyIntervention(multiproduct)
    
    def test_backend_error_handling(self):
        """Test backend switching error handling."""
        # Test invalid backend
        with pytest.raises(ValueError, match="Unknown backend"):
            use_backend("nonexistent_backend")
        
        # Test empty/None backend
        with pytest.raises((ValueError, TypeError)):
            use_backend("")
    
    def test_score_method_edge_cases(self):
        """Test model score method with edge cases."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        t = [1, 2, 3, 4, 5]
        
        # Perfect predictions should give R² = 1
        y_perfect = model.predict(t)
        score = model.score(t, y_perfect)
        assert np.isclose(score, 1.0, atol=1e-6)
        
        # Constant predictions vs varying data (poor fit)
        y_varying = [10, 30, 50, 70, 90]
        
        # Create a model that always predicts the same value
        constant_model = BassModel()
        constant_model.params_ = {"p": 0, "q": 0, "m": 50}  # Will predict ~0 always
        
        score_poor = constant_model.score(t, y_varying)
        assert score_poor <= 0.1  # Should be very poor fit
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        model = BassModel()
        
        # Parameters at machine precision limits
        model.params_ = {
            "p": np.finfo(float).eps,
            "q": np.finfo(float).eps, 
            "m": 1.0
        }
        
        t = [1, 2, 3]
        result = model.predict(t)
        assert np.all(np.isfinite(result))
        
        # Very large time values (test saturation)
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        t_large = [100, 1000, 10000]
        result_large = model.predict(t_large)
        
        # Should saturate near market potential
        assert np.all(result_large <= model.params_["m"] + 1e-6)
        assert np.all(np.isfinite(result_large))
    
    def test_array_type_consistency(self):
        """Test consistency across different input array types."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Test with list, numpy array, and pandas Series
        t_list = [1, 2, 3, 4, 5]
        t_array = np.array(t_list)
        t_series = pd.Series(t_list)
        
        result_list = model.predict(t_list)
        result_array = model.predict(t_array)
        result_series = model.predict(t_series)
        
        # Results should be equivalent
        np.testing.assert_array_almost_equal(result_list, result_array)
        np.testing.assert_array_almost_equal(result_list, result_series)
    
    def test_parameter_bounds_validation(self):
        """Test parameter boundary conditions."""
        model = BassModel()
        
        # Test with p=0 (no innovation effect)
        model.params_ = {"p": 0, "q": 0.3, "m": 1000}
        result = model.predict([1, 2, 3, 4, 5])
        assert np.all(np.isfinite(result))
        
        # Test with q=0 (no imitation effect)
        model.params_ = {"p": 0.02, "q": 0, "m": 1000}
        result = model.predict([1, 2, 3, 4, 5])
        assert np.all(np.isfinite(result))
    
    def test_large_scale_performance(self):
        """Test performance with large input sizes."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Large time array (10,000 points)
        t_large = np.arange(1, 10001)
        result = model.predict(t_large)
        
        assert len(result) == len(t_large)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.all(result <= model.params_["m"] + 1e-6)


class TestModelSpecificEdgeCases:
    """Edge cases specific to individual model types."""
    
    def test_logistic_model_edge_cases(self):
        """Test LogisticModel specific edge cases."""
        model = LogisticModel()
        
        # Test with extreme growth rate
        model.params_ = {"L": 1000, "k": 100, "x0": 5}
        t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = model.predict(t)
        
        # Should saturate quickly with high k
        assert np.all(np.isfinite(result))
        assert result[-1] >= result[0]  # Should be non-decreasing
        
        # Test with zero growth rate
        model.params_ = {"L": 1000, "k": 0, "x0": 5}
        result_zero_k = model.predict(t)
        
        # With k=0, should be constant at L/2
        expected_constant = model.params_["L"] / 2
        assert np.allclose(result_zero_k, expected_constant)
    
    def test_gompertz_model_edge_cases(self):
        """Test GompertzModel specific edge cases."""
        model = GompertzModel()
        
        # Test with extreme parameters
        model.params_ = {"a": 1000, "b": 100, "c": 0.99}
        t = [1, 2, 3, 4, 5]
        result = model.predict(t)
        
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.all(result <= model.params_["a"])
        
        # Test with c approaching 1 (slow decay)
        model.params_ = {"a": 1000, "b": 2, "c": 0.999}
        result_slow = model.predict(t)
        assert np.all(np.isfinite(result_slow))
    
    def test_bass_model_parameter_combinations(self):
        """Test Bass model with various parameter combinations."""
        model = BassModel()
        t = [1, 2, 3, 4, 5]
        
        # High innovation, low imitation
        model.params_ = {"p": 0.1, "q": 0.01, "m": 1000}
        result1 = model.predict(t)
        
        # Low innovation, high imitation  
        model.params_ = {"p": 0.001, "q": 0.8, "m": 1000}
        result2 = model.predict(t)
        
        # Both should be valid
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))
        assert np.all(result1 >= 0)
        assert np.all(result2 >= 0)
        
        # Different adoption patterns should emerge
        assert not np.allclose(result1, result2, rtol=0.1)


class TestErrorRecovery:
    """Test error recovery and graceful handling."""
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero scenarios."""
        model = BassModel()
        
        # Very small p value (potential division by zero in Bass formula)
        model.params_ = {"p": 1e-15, "q": 0.3, "m": 1000}
        t = [1, 2, 3]
        
        try:
            result = model.predict(t)
            assert np.all(np.isfinite(result))
        except (ZeroDivisionError, RuntimeError, ValueError):
            # If the implementation doesn't handle this, exception is acceptable
            pass
    
    def test_overflow_protection(self):
        """Test protection against numerical overflow."""
        model = BassModel()
        model.params_ = {"p": 100, "q": 100, "m": 1e10}
        
        # Very large parameters might cause overflow
        t = [1, 2, 3]
        result = model.predict(t)
        
        # Should either handle gracefully or saturate at market potential
        assert np.all(np.isfinite(result) | (result <= model.params_["m"]))
    
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # NaN in time array
        t_with_nan = [1, 2, np.nan, 4, 5]
        
        try:
            result = model.predict(t_with_nan)
            # If it doesn't raise an error, should handle NaN appropriately
            assert len(result) == len(t_with_nan)
        except (ValueError, RuntimeError):
            # Raising an error for NaN input is also acceptable
            pass
    
    def test_infinite_input_handling(self):
        """Test handling of infinite inputs."""
        model = BassModel()
        model.params_ = {"p": 0.02, "q": 0.3, "m": 1000}
        
        # Infinite time values
        t_with_inf = [1, 2, np.inf, 4, 5]
        
        try:
            result = model.predict(t_with_inf)
            # Should handle infinite inputs gracefully
            assert len(result) == len(t_with_inf)
            # At infinite time, should approach market potential
            if np.isfinite(result[2]):
                assert result[2] <= model.params_["m"] + 1e-6
        except (ValueError, RuntimeError, OverflowError):
            # Raising an error for infinite input is also acceptable
            pass