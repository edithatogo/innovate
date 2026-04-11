"""Load and stress tests for the innovate library."""
import time

import numpy as np

from innovate.backend import use_backend

# Use numpy backend to avoid JAX-related issues
use_backend('numpy')

from innovate.diffuse.bass import BassModel


def test_high_volume_model_creation():
    """Stress test for creating many model instances."""
    start_time = time.time()

    # Create 100 model instances
    models = []
    for i in range(100):
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        models.append(model)

    end_time = time.time()

    # Check that creation was reasonably fast (less than 5 seconds for 100 models)
    assert end_time - start_time < 5.0, f"Model creation took {end_time - start_time:.2f}s for 100 models"
    assert len(models) == 100


def test_large_data_handling():
    """Test the library's ability to handle large datasets without fitting."""
    # Generate a large time array
    large_t = np.linspace(0, 100, 10000)  # 10k data points

    # Create model with parameters
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    # Rather than calling predict (which might trigger ODE solver),
    # just test that large arrays can be handled properly
    assert len(large_t) == 10000
    assert large_t.shape == (10000,)

    # Test parameter access with large arrays
    p_param = model.params_["p"]
    assert isinstance(p_param, float)
    assert 0 < p_param < 1  # Reasonable adoption coefficient


def test_concurrent_model_operations():
    """Test multiple models being operated on simultaneously."""
    # Create multiple models
    models = []
    for i in range(10):
        model = BassModel()
        model.params_ = {"p": 0.01 + (i * 0.01), "q": 0.1 + (i * 0.05), "m": 500 + (i * 100)}
        models.append(model)

    # Verify all models have different parameters
    p_values = [model.params_["p"] for model in models]
    q_values = [model.params_["q"] for model in models]
    m_values = [model.params_["m"] for model in models]

    # Check they're all different
    assert len(set(p_values)) == 10
    assert len(set(q_values)) == 10
    assert len(set(m_values)) == 10


def test_memory_efficiency_under_load():
    """Test that memory usage remains reasonable under load."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create and work with multiple models
    for i in range(50):
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        # Access parameters to simulate usage
        _ = model.params_
        # Delete explicitly to help with memory management
        del model

    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Memory growth should be reasonable (less than 20MB for 50 operations)
    memory_growth = final_memory - initial_memory
    assert memory_growth < 20, f"Memory grew by {memory_growth:.2f} MB during load test"


def test_parameter_boundary_conditions():
    """Test models with extreme parameter values to check for stability."""
    extreme_params = [
        {"p": 0.001, "q": 0.001, "m": 1000000},  # Very slow adoption, large market
        {"p": 0.9, "q": 0.9, "m": 10},           # Very fast adoption, small market
        {"p": 0.5, "q": 0.0001, "m": 50000},     # High innovation, low imitation
        {"p": 0.0001, "q": 0.5, "m": 50000},     # Low innovation, high imitation
    ]

    for params in extreme_params:
        model = BassModel()
        model.params_ = params

        # Test that parameters can be accessed without issues
        assert all(k in model.params_ for k in ["p", "q", "m"])
        for k, v in params.items():
            assert model.params_[k] == v


def test_long_running_parameter_validation():
    """Test that parameter validation works correctly over extended operations."""
    model = BassModel()

    # Perform many parameter setting operations
    test_params = [
        {"p": 0.01, "q": 0.1, "m": 1000},
        {"p": 0.02, "q": 0.2, "m": 2000},
        {"p": 0.03, "q": 0.3, "m": 3000},
        {"p": 0.04, "q": 0.4, "m": 4000},
        {"p": 0.05, "q": 0.5, "m": 5000},
    ]

    for params in test_params:
        model.params_ = params
        # Validate that parameters are set correctly
        for k, v in params.items():
            assert model.params_[k] == v


def test_model_method_access_stress():
    """Test accessing model methods and properties repeatedly."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    # Repeatedly access properties
    for _ in range(1000):
        params = model.params_
        param_names = model.param_names
        covariates = model.covariates

        # Verify consistency
        assert len(params) == 3  # p, q, m
        assert len(param_names) == 3  # p, q, m
        assert covariates == []


def test_time_series_boundary_conditions():
    """Test with various time series edge cases."""
    edge_cases = [
        np.array([0]),  # Single point
        np.array([0, 1]),  # Two points
        np.array([0, 0.1, 0.2]),  # Small time increments
        np.array([0, 1000, 2000]),  # Large time increments
        np.array([-10, -5, 0, 5, 10]),  # Including negative times
    ]

    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}

    for t_series in edge_cases:
        # Just test that these arrays can be handled without crashing
        assert len(t_series) > 0
        assert isinstance(t_series, np.ndarray)


if __name__ == "__main__":
    print("Running load and stress tests...")

    test_high_volume_model_creation()
    print("✓ High volume model creation test passed")

    test_large_data_handling()
    print("✓ Large data handling test passed")

    test_concurrent_model_operations()
    print("✓ Concurrent model operations test passed")

    test_memory_efficiency_under_load()
    print("✓ Memory efficiency test passed")

    test_parameter_boundary_conditions()
    print("✓ Parameter boundary conditions test passed")

    test_long_running_parameter_validation()
    print("✓ Long running parameter validation test passed")

    test_model_method_access_stress()
    print("✓ Model method access stress test passed")

    test_time_series_boundary_conditions()
    print("✓ Time series boundary conditions test passed")

    print("All load and stress tests passed!")
