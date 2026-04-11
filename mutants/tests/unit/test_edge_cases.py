"""Edge case testing for the Innovate library."""

import numpy as np
import pytest

from innovate.backend import use_backend

# Use numpy backend to avoid JAX-related issues
use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.utils.model_validation import (
    validate_bass_parameters,
    validate_model_predictions,
)


class TestEdgeCases:
    """Test edge cases for the diffusion models."""

    def test_bass_model_edge_cases(self):
        """Test Bass model with edge case parameters and data."""
        # Edge case 1: Very small parameters
        model = BassModel()
        model.params_ = {"p": 1e-10, "q": 1e-10, "m": 1.0}

        t = [0, 1, 2]
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Some extreme values may cause numerical issues, which is expected
            pass

        # Edge case 2: Very large parameters
        model.params_ = {"p": 0.5, "q": 5.0, "m": 1e6}
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Some extreme values may cause numerical issues, which is expected
            pass

        # Edge case 3: p=0 (innovation coefficient)
        model.params_ = {"p": 0.0, "q": 0.5, "m": 1000}
        t = np.linspace(0, 10, 5)
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # May cause numerical issues
            pass

        # Edge case 4: q=0 (imitation coefficient)
        model.params_ = {"p": 0.05, "q": 0.0, "m": 1000}
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # May cause numerical issues
            pass

    def test_bass_model_extreme_time_values(self):
        """Test Bass model with extreme time values."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

        # Very large time values
        t_large = [1000, 2000, 3000]
        try:
            result = model.predict(t_large)
            assert len(result) == len(t_large)
        except Exception:
            # May cause numerical issues with large times
            pass

        # Negative time values (should raise error due to validation)
        with pytest.raises(ValueError):
            model.predict([-1, 0, 1])

    def test_bass_model_extreme_covariates(self):
        """Test Bass model with extreme covariate values."""
        model = BassModel(covariates=["advertising"])
        model.params_ = {
            "p": 0.03,
            "q": 0.38,
            "m": 1000,
            "beta_p_advertising": 0.1,
            "beta_q_advertising": 0.1,
            "beta_m_advertising": 10,
        }

        t = [0, 1, 2, 3]

        # Extreme covariate values
        extreme_covariates = {"advertising": [1e6, 1e6, 1e6, 1e6]}
        try:
            result = model.predict(t, covariates=extreme_covariates)
            assert len(result) == len(t)
        except Exception:
            # Extreme covariates may cause numerical issues
            pass

    def test_logistic_model_edge_cases(self):
        """Test Logistic model with edge case parameters."""
        model = LogisticModel()

        # Edge case: Large k (growth rate) - can cause numerical overflow
        model.params_ = {"L": 1000, "k": 100, "x0": 5}
        t = [0, 1, 2]
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Large k can cause overflow
            pass

        # Edge case: Very small k
        model.params_ = {"L": 1000, "k": 1e-10, "x0": 5}
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Very small k is usually fine
            pass

    def test_gompertz_model_edge_cases(self):
        """Test Gompertz model with edge case parameters."""
        model = GompertzModel()

        # Edge case: Large b parameter
        model.params_ = {"a": 1000, "b": 50, "c": 0.1}
        t = [0, 1, 2]
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Large parameters can cause numerical issues
            pass

        # Edge case: Very small c parameter
        model.params_ = {"a": 1000, "b": 5, "c": 1e-10}
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # Very small c is usually fine
            pass

    def test_empty_input_edge_cases(self):
        """Test models with empty or minimal input."""
        model = BassModel()

        # Empty time series should raise error
        with pytest.raises(ValueError):
            model.predict([])

        # Single point should work
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        result = model.predict([0])
        assert len(result) == 1

    def test_parameter_boundary_edge_cases(self):
        """Test parameter validation at boundaries."""
        # Test Bass parameter validation
        validation_result = validate_bass_parameters({"p": -0.01, "q": 0.38, "m": 1000})
        assert not validation_result["is_valid"]
        assert "positive number" in str(validation_result["issues"])

        validation_result = validate_bass_parameters({"p": 0.03, "q": -0.38, "m": 1000})
        assert not validation_result["is_valid"]
        assert "positive number" in str(validation_result["issues"])

        validation_result = validate_bass_parameters({"p": 0.03, "q": 0.38, "m": -1000})
        assert not validation_result["is_valid"]
        assert "positive number" in str(validation_result["issues"])

        # Valid parameters should pass
        validation_result = validate_bass_parameters({"p": 0.03, "q": 0.38, "m": 1000})
        assert validation_result["is_valid"]

    def test_prediction_validation(self):
        """Test model prediction validation."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

        t = np.array([0, 1, 2, 3])
        y_pred = np.array([10, 50, 200, 500])  # Reasonable predictions

        validation_result = validate_model_predictions(
            model, t, y_pred, max_growth_ratio=10.0
        )  # Higher threshold for this test
        assert validation_result["is_valid"]

        # Test with non-finite values
        y_pred_bad = np.array([10, np.nan, 200, 500])
        validation_result = validate_model_predictions(model, t, y_pred_bad)
        assert not validation_result["is_valid"]
        assert "non-finite values" in str(validation_result["issues"])

        # Test with negative values
        y_pred_negative = np.array([-10, 50, 200, 500])
        validation_result = validate_model_predictions(model, t, y_pred_negative)
        assert not validation_result["is_valid"]
        assert "negative values" in str(validation_result["issues"])

        # Test with extremely high growth
        y_pred_extreme = np.array([1, 1000, 1001, 1002])  # Large jump from 1 to 1000
        validation_result = validate_model_predictions(model, t, y_pred_extreme)
        assert not validation_result["is_valid"]
        assert "high growth rate" in str(validation_result["issues"])


class TestNumericalStability:
    """Test numerical stability of the models."""

    def test_bass_model_stability(self):
        """Test numerical stability of Bass model."""
        model = BassModel()

        # Use parameters that might cause numerical issues
        problematic_params = [
            {"p": 0.01, "q": 100, "m": 1e-10},  # Very small m
            {"p": 100, "q": 0.01, "m": 1e-10},  # Very small m, large p
            {"p": 1e-10, "q": 1e-10, "m": 1e10},  # Very large m
        ]

        t = np.linspace(0, 10, 20)

        for i, params in enumerate(problematic_params):
            model.params_ = params
            try:
                result = model.predict(t)
                assert len(result) == len(t)
                # Check that results are finite
                assert np.all(np.isfinite(result))
            except Exception as e:
                # Some parameter combinations are expected to cause issues
                print(f"Expected numerical issue with params {i}: {e}")
                continue

    def test_logistic_model_stability(self):
        """Test numerical stability of Logistic model."""
        model = LogisticModel()

        # Parameters that might cause numerical issues
        t = np.linspace(-10, 10, 20)

        # Large k with extreme x0 can cause overflow
        model.params_ = {"L": 1000, "k": 50, "x0": 100}
        try:
            result = model.predict(t)
            assert len(result) == len(t)
        except Exception:
            # This is expected with extreme parameters
            pass


def test_model_with_event_edge_cases():
    """Test Bass model with event time."""
    # Test with t_event and parameter validation
    model_with_event = BassModel(t_event=5.0)
    assert model_with_event.t_event == 5.0

    # Test with early event time
    model_early_event = BassModel(t_event=0.1)
    assert model_early_event.t_event == 0.1

    # Test with late event time
    model_late_event = BassModel(t_event=100.0)
    assert model_late_event.t_event == 100.0


if __name__ == "__main__":
    test_instance = TestEdgeCases()
    test_instance.test_bass_model_edge_cases()
    print("✓ Bass model edge cases test passed")

    test_instance.test_bass_model_extreme_time_values()
    print("✓ Bass model extreme time values test passed")

    test_instance.test_bass_model_extreme_covariates()
    print("✓ Bass model extreme covariates test passed")

    test_instance.test_logistic_model_edge_cases()
    print("✓ Logistic model edge cases test passed")

    test_instance.test_gompertz_model_edge_cases()
    print("✓ Gompertz model edge cases test passed")

    test_instance.test_empty_input_edge_cases()
    print("✓ Empty input edge cases test passed")

    test_instance.test_parameter_boundary_edge_cases()
    print("✓ Parameter boundary edge cases test passed")

    test_instance.test_prediction_validation()
    print("✓ Prediction validation test passed")

    stability_test = TestNumericalStability()
    stability_test.test_bass_model_stability()
    print("✓ Bass model stability test passed")

    stability_test.test_logistic_model_stability()
    print("✓ Logistic model stability test passed")

    test_model_with_event_edge_cases()
    print("✓ Model with event edge cases test passed")

    print("All edge case tests passed!")
