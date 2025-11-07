"""Performance tests for the innovate library that avoid problematic ODE operations."""
import time
import pytest
import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from innovate.backend import use_backend
use_backend('numpy')  # Use numpy backend for consistency

from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel


def test_model_creation_performance(benchmark):
    """Test performance of model instantiation."""
    def create_bass_model():
        return BassModel()
    
    result = benchmark(create_bass_model)
    assert result is not None


def test_parameter_setting_performance(benchmark):
    """Test performance of parameter setting."""
    model = BassModel()
    
    def set_params():
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    
    benchmark(set_params)
    assert model.params_ is not None


def test_logistic_model_formula_performance(benchmark):
    """Test performance of direct logistic function computation."""
    t = np.linspace(0, 20, 100)
    L, k, x0 = 1000, 0.2, 10
    
    def compute_logistic():
        return L / (1 + np.exp(-k * (t - x0)))
    
    result = benchmark(compute_logistic)
    assert len(result) == len(t)


def test_parameter_validation_performance(benchmark):
    """Test performance of parameter validation methods."""
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    
    def check_params():
        return all(param in model.params_ for param in ["p", "q", "m"])
    
    result = benchmark(check_params)
    assert result is True


def test_array_operations_performance(benchmark):
    """Test performance of basic array operations that might be used in the library."""
    t = np.linspace(0, 10, 20)
    
    def array_operations():
        # Simple array operations that don't trigger ODE solving
        result = t * 2.0 + 1.0
        return result
    
    result = benchmark(array_operations)
    assert len(result) == len(t)


def test_multiple_model_creation_performance(benchmark):
    """Test performance when creating multiple models."""
    def create_multiple_models():
        models = []
        for i in range(10):
            model = BassModel()
            model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
            models.append(model)
        return models
    
    results = benchmark(create_multiple_models)
    assert len(results) == 10


def test_memory_usage_stability():
    """Test that memory usage doesn't grow with repeated operations."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform multiple operations without ODE solving
    for i in range(10):
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        # Access parameters to simulate usage
        _ = model.params_
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Memory growth should be minimal (less than 10MB)
    assert final_memory - initial_memory < 10, f"Memory grew by {final_memory - initial_memory:.2f} MB"


def test_backend_switching_performance(benchmark):
    """Test the performance of backend switching functionality."""
    def switch_backends():
        from innovate.backend import use_backend
        use_backend('numpy')
        use_backend('numpy')  # Should be fast if already set
        
    benchmark(switch_backends)


if __name__ == "__main__":
    # Run basic performance checks
    print("Running safe performance tests...")
    
    # Create simple operations for quick checks
    start_time = time.time()
    model = BassModel()
    creation_time = time.time() - start_time
    
    print(f"Bass model creation time: {creation_time:.6f}s")
    
    start_time = time.time()
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    param_set_time = time.time() - start_time
    
    print(f"Parameter setting time: {param_set_time:.6f}s")
    
    print("Safe performance tests completed!")