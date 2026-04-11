"""Tests for the optimization utilities."""
from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel
from src.innovate.utils.optimization_guide import (
    benchmark_model_performance,
    suggest_parameter_bounds_safely,
    validate_parameters_safely,
)


def test_optimization_utilities():
    """Test the optimization utilities."""
    # Test benchmarking
    params = {"p": 0.03, "q": 0.38, "m": 1000}
    t_data = [0, 1, 2, 3, 4]

    results = benchmark_model_performance(BassModel, params, t_data)
    assert "param_validation_time" in results
    assert "param_count" in results
    assert results["param_count"] == 3  # p, q, m

    # Test parameter validation
    valid = validate_parameters_safely(BassModel, params)
    assert valid is True

    # Test bounds suggestion
    y_data = [10, 20, 30, 40, 50]
    bounds = suggest_parameter_bounds_safely(BassModel, y_data)
    assert "m" in bounds
    assert bounds["m"][0] > 0

    # Test with logistic model
    logistic_params = {"L": 1000, "k": 0.2, "x0": 10}
    results_logistic = benchmark_model_performance(LogisticModel, logistic_params, t_data)
    assert results_logistic["param_count"] == 3  # L, k, x0


def test_parameter_validation_edge_cases():
    """Test parameter validation with edge cases."""
    # Test with negative parameters (should warn but not error)
    params_with_negative = {"p": -0.01, "q": 0.38, "m": 1000}
    valid = validate_parameters_safely(BassModel, params_with_negative)
    assert valid is True


def test_bounds_suggestion():
    """Test bounds suggestion with different data."""
    # Test with empty data
    empty_bounds = suggest_parameter_bounds_safely(BassModel, [])
    assert empty_bounds == {}

    # Test with single value
    single_bounds = suggest_parameter_bounds_safely(BassModel, [50])
    assert "m" in single_bounds
    assert single_bounds["m"][0] == 50


if __name__ == "__main__":
    print("Testing optimization utilities...")

    test_optimization_utilities()
    print("✓ Basic optimization utilities test passed")

    test_parameter_validation_edge_cases()
    print("✓ Parameter validation edge cases test passed")

    test_bounds_suggestion()
    print("✓ Bounds suggestion test passed")

    print("All optimization utility tests passed!")
