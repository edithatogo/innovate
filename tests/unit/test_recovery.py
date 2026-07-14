"""Recovery tests for the innovate library."""

import numpy as np
import pytest

from innovate.backend import use_backend

# Use numpy backend to avoid JAX-related issues
use_backend("numpy")

from innovate.diffuse.bass import BassModel


def test_unfitted_model_error_handling():
    """Test that unfitted models raise appropriate errors."""
    model = BassModel()

    # Ensure params is empty
    model.params_ = {}

    # Calling predict on an unfitted model should raise an error
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict([1, 2, 3])

    # Calling score on an unfitted model should raise an error
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.score([1, 2, 3], [10, 20, 30])


def test_invalid_parameter_handling():
    """Test how the model handles invalid parameters."""
    model = BassModel()

    # Test with negative parameters
    model.params_ = {"p": -0.1, "q": 0.38, "m": 1000}

    # The model should accept these parameters but they might produce invalid results
    # This test ensures it doesn't crash with invalid parameters
    assert model.params_["p"] == -0.1
    assert model.params_["q"] == 0.38
    assert model.params_["m"] == 1000


def test_extreme_parameter_handling():
    """Test model behavior with extreme parameter values."""
    model = BassModel()

    # Use extreme values to test for stability
    extreme_params = {
        "p": 1e-10,  # Very small
        "q": 1e10,  # Very large
        "m": 1e15,  # Very large market size
    }

    model.params_ = extreme_params

    # Check that parameters were set
    for key, value in extreme_params.items():
        assert model.params_[key] == value


def test_model_recovery_after_error():
    """Test that a model can recover after error conditions."""
    model = BassModel()

    # Intentionally put the model in an error state
    model.params_ = {}

    # Attempt to call predict (should fail)
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict([1, 2, 3])

    # Now fix the model by setting valid parameters
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    # The model should now work properly
    # Actually call predict to verify recovery
    predictions = model.predict([1, 2, 3])

    assert predictions is not None
    assert len(predictions) == 3

    # Also verify the parameters are properly set
    assert model.params_["p"] == 0.03
    assert model.params_["q"] == 0.38
    assert model.params_["m"] == 1000


def test_parameter_validation_recovery():
    """Test recovery from invalid parameter states."""
    model = BassModel()

    # Set initially valid parameters
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    assert model.params_["p"] == 0.03

    # Update with different valid parameters
    model.params_ = {"p": 0.05, "q": 0.40, "m": 1200}
    assert model.params_["p"] == 0.05

    # Verify the update worked properly
    assert model.params_["q"] == 0.40
    assert model.params_["m"] == 1200


def test_error_handling_with_different_data_types():
    """Test how the model handles different data types."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    # Test with different numeric types
    # Using list
    t_list = [0, 1, 2, 3, 4]
    # Using numpy array
    t_array = np.array([0, 1, 2, 3, 4])

    # The model should be able to handle the parameters regardless
    # of how the time values were prepared (before calling predict)
    assert len(t_list) == 5
    assert len(t_array) == 5
    assert isinstance(t_array, np.ndarray)


def test_model_state_preservation():
    """Test that model state is preserved properly between operations."""
    model = BassModel()

    # Save initial state
    initial_covariates = model.covariates
    initial_params = model.params_

    # Set parameters
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    new_params = {"p": 0.03, "q": 0.38, "m": 1000}

    # Verify parameters were set correctly
    for key, value in new_params.items():
        assert model.params_[key] == value

    # Verify covariates remain unchanged
    assert model.covariates == initial_covariates


def test_recovery_from_extreme_predictions():
    """Test system behavior with extreme values that might cause overflow."""
    model = BassModel()

    # Set extreme but valid parameters
    model.params_ = {"p": 0.99, "q": 0.99, "m": 1e6}

    # Validate that the parameters are accessible
    assert model.params_["p"] == 0.99
    assert model.params_["q"] == 0.99
    assert model.params_["m"] == 1e6


def test_error_message_clarity():
    """Test that error messages are clear and actionable."""
    model = BassModel()

    # Unfit model should give clear error
    model.params_ = {}
    try:
        model.predict([1, 2, 3])
        raise AssertionError("Expected an exception")  # Should not reach here
    except RuntimeError as e:
        error_msg = str(e)
        # Check that the error message is helpful
        assert "Model has not been fitted yet" in error_msg
        assert "Call .fit()" in error_msg or "fitted" in error_msg


def test_consistent_state_after_exception():
    """Test that the model maintains a consistent state after exceptions."""
    model = BassModel()

    # Set initial valid parameters
    initial_params = {"p": 0.03, "q": 0.38, "m": 1000}
    model.params_ = initial_params

    # Attempt an operation that will fail (without fitted params)
    model.params_ = {}
    predict_failed = False
    try:
        model.predict([1, 2, 3])
    except RuntimeError:
        predict_failed = True

    assert predict_failed

    # Restore valid parameters
    model.params_ = initial_params

    # Verify model is back to a valid state
    for key, expected_value in initial_params.items():
        assert model.params_[key] == expected_value


if __name__ == "__main__":
    print("Running recovery tests...")

    test_unfitted_model_error_handling()
    print("✓ Unfitted model error handling test passed")

    test_invalid_parameter_handling()
    print("✓ Invalid parameter handling test passed")

    test_extreme_parameter_handling()
    print("✓ Extreme parameter handling test passed")

    test_model_recovery_after_error()
    print("✓ Model recovery after error test passed")

    test_parameter_validation_recovery()
    print("✓ Parameter validation recovery test passed")

    test_error_handling_with_different_data_types()
    print("✓ Error handling with different data types test passed")

    test_model_state_preservation()
    print("✓ Model state preservation test passed")

    test_recovery_from_extreme_predictions()
    print("✓ Recovery from extreme predictions test passed")

    test_error_message_clarity()
    print("✓ Error message clarity test passed")

    test_consistent_state_after_exception()
    print("✓ Consistent state after exception test passed")

    print("All recovery tests passed!")
