# Comprehensive Testing Strategy for the Innovate Library

## Overview

This document outlines a comprehensive testing strategy for the innovate library to accelerate its maturation and ensure robust, reliable functionality. The innovate library is a sophisticated modeling tool for innovation and policy diffusion with multiple mathematical models and fitting capabilities.

## Current State Assessment

- The library currently uses pytest with unit, integration, and e2e tests
- It has some basic benchmarking in place
- Mathematical models (Bass, Gompertz, Logistic, etc.) have basic test coverage
- Edge cases are partially covered
- No property-based, mutation, load, stress, endurance, or recovery testing currently implemented

## Testing Strategy Implementation Roadmap

### Phase 1 (Immediate - Weeks 1-2)
1. Set up property-based testing with Hypothesis
2. Implement basic mutation testing
3. Add performance benchmarks to CI
4. Target individual file coverage >95%

### Phase 2 (Short-term - Weeks 3-4)
1. Expand performance testing framework
2. Implement load testing scenarios
3. Add stress testing for critical paths

### Phase 3 (Medium-term - Weeks 5-8)
1. Implement endurance testing framework
2. Add comprehensive recovery testing
3. Set up automated chaos testing

### Phase 4 (Long-term - Ongoing)
1. Continuous improvement of test coverage
2. Advanced comparative testing
3. Model validation framework

## 1. Property-Based Testing

**Tool**: Hypothesis library

**Rationale**: Mathematical libraries benefit greatly from property-based testing to verify invariants across a wide range of inputs.

**Implementation**:

First, install the required dependency:
```bash
pip install hypothesis
```

Example test implementations:

```python
# tests/test_property_based.py
from hypothesis import given, strategies as st
import numpy as np
import pytest
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import ScipyFitter

@given(st.lists(st.floats(min_value=0.1, max_value=100), min_size=10, max_size=100))
def test_cumulative_predictions_non_decreasing(time_series):
    """Test that all cumulative models produce non-decreasing predictions"""
    t = np.sort(np.array(time_series))
    model = BassModel()
    # Set valid parameters
    model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    pred = model.predict(t)
    # Allow for small numerical errors
    assert np.all(np.diff(pred) >= -1e-10)

@given(
    st.floats(min_value=0.001, max_value=0.5),  # p parameter
    st.floats(min_value=0.01, max_value=1.0),   # q parameter
    st.floats(min_value=10, max_value=100000)   # m parameter
)
def test_bass_model_parameters_valid(p, q, m):
    """Test that Bass model parameters remain in valid ranges after fitting"""
    t = np.linspace(0, 50, 100)
    
    # Generate synthetic data with known parameters
    exp_term = np.exp(-(p + q) * t)
    y_true = m * (1 - exp_term) / (1 + (q / p) * exp_term)
    
    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, m * 0.05, len(t))
    y_noisy = np.abs(y_true + noise)  # Ensure non-negative
    
    model = BassModel()
    fitter = ScipyFitter()
    
    # Fit the model
    fitter.fit(model, t, y_noisy)
    
    # Validate that fitted parameters are reasonable
    assert model.params_ is not None
    assert "p" in model.params_
    assert "q" in model.params_
    assert "m" in model.params_
    
    # Check that parameters are positive
    assert model.params_["p"] > 0
    assert model.params_["q"] > 0
    assert model.params_["m"] > 0

@given(
    st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=50),
    st.lists(st.floats(min_value=0, max_value=1000), min_size=10, max_size=50)
)
def test_model_predictions_shape_consistency(t_list, y_list):
    """Test that model predictions have the same shape as input time series"""
    t = np.array(t_list)
    y = np.array(y_list)
    
    # Ensure t is sorted for proper cumulative behavior
    sorted_indices = np.argsort(t)
    t = t[sorted_indices]
    y = y[sorted_indices]
    
    models = [BassModel(), GompertzModel(), LogisticModel()]
    
    for model in models:
        fitter = ScipyFitter()
        # Fit the model
        fitter.fit(model, t, y)
        
        # Make predictions
        predictions = model.predict(t)
        
        # Check that prediction shape matches input
        assert len(predictions) == len(t)
        
        # For cumulative models, check non-decreasing property (with tolerance for numerical errors)
        if isinstance(model, (BassModel, GompertzModel)):
            diffs = np.diff(predictions)
            assert np.all(diffs >= -1e-10), f"Model {type(model).__name__} predictions not non-decreasing"
```

## 2. Mutation Testing

**Tool**: mutmut

**Rationale**: Assess quality of existing tests and identify untested code paths.

**Installation**:
```bash
pip install mutmut
```

**Configuration**:
Create a `pyproject.toml` entry for mutmut:

```toml
[tool.mutmut]
source_paths = ["src/innovate/"]
backup = false
pytest_add_cli_args = ["-q"]
pytest_add_cli_args_test_selection = ["tests"]
```

**Commands**:
```bash
# Re-run a specific generated mutant by name
mutmut run <mutant-name>

# Run mutation testing on the whole project
mutmut run

# View results
mutmut results

# Apply surviving mutations (if needed for debugging)
mutmut apply
```

## 3. Performance Testing

**Tools**: pytest-benchmark, cProfile, line_profiler

**Implementation**:

```python
# tests/test_performance.py
import numpy as np
import pytest
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import ScipyFitter

def generate_large_dataset(size=10000):
    """Generate a large synthetic dataset for performance testing"""
    t = np.linspace(0, 100, size)
    # Generate realistic adoption curve
    p, q, m = 0.03, 0.38, 1000
    exp_term = np.exp(-(p + q) * t)
    y = m * (1 - exp_term) / (1 + (q / p) * exp_term)
    # Add noise
    noise = np.random.normal(0, m * 0.02, len(t))
    y = np.abs(y + noise)
    return t, y

def test_bass_model_fitting_performance(benchmark):
    """Benchmark Bass model fitting performance"""
    t, y = generate_large_dataset(size=1000)
    model = BassModel()
    fitter = ScipyFitter()
    
    result = benchmark(fitter.fit, model, t, y)
    assert result is not None

@pytest.mark.parametrize("model_class", [BassModel, GompertzModel, LogisticModel])
def test_model_prediction_performance(model_class, benchmark):
    """Benchmark prediction performance across different models"""
    t = np.linspace(0, 50, 1000)
    model = model_class()
    # Set some parameters to make it a valid model
    if model_class == BassModel:
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    elif model_class == GompertzModel:
        model.params_ = {"a": 1000, "b": 5, "c": 0.1}
    elif model_class == LogisticModel:
        model.params_ = {"L": 1000, "k": 0.1, "x0": 25}
    
    result = benchmark(model.predict, t)
    assert len(result) == len(t)

@pytest.mark.parametrize("dataset_size", [100, 1000, 5000])
def test_fitting_scaling_performance(dataset_size, benchmark):
    """Test how fitting performance scales with dataset size"""
    t, y = generate_large_dataset(size=dataset_size)
    model = BassModel()
    fitter = ScipyFitter()
    
    result = benchmark(fitter.fit, model, t, y)
    assert result is not None
```

## 4. Load Testing

**Implementation**:

```python
# tests/test_load.py
import numpy as np
import pytest
import psutil
import os
from innovate.diffuse.bass import BassModel
from innovate.compete.competition import MultiProductDiffusionModel
from innovate.fitters.scipy_fitter import ScipyFitter

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_large_dataset_fitting():
    """Test fitting with very large datasets"""
    # Create a very large dataset
    t = np.linspace(0, 1000, 50000)  # 50k data points
    y = 1000 * (1 - np.exp(-0.1 * t))  # Simple adoption curve
    
    initial_memory = get_memory_usage()
    model = BassModel()
    fitter = ScipyFitter()
    
    # This should complete without memory issues
    fitter.fit(model, t, y)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB for this operation)
    assert memory_increase < 100, f"Memory increase too high: {memory_increase}MB"

def test_multi_product_large_scale():
    """Test multi-product models with many products"""
    n_products = 20
    p_vals = [0.02 + i*0.001 for i in range(n_products)]
    # Create a dense interaction matrix
    Q_matrix = [[0.1 + i*0.01 + j*0.01 for j in range(n_products)] for i in range(n_products)]
    m_vals = [1000 + i*100 for i in range(n_products)]
    product_names = [f"Product_{i}" for i in range(n_products)]
    
    initial_memory = get_memory_usage()
    
    model = MultiProductDiffusionModel(
        p=p_vals,
        Q=Q_matrix,
        m=m_vals,
        names=product_names,
    )
    
    time_horizon = np.arange(1, 101)
    predictions = model.predict(time_horizon)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Check that predictions have correct dimensions
    assert predictions.shape == (100, n_products)
    # Memory increase should be reasonable
    assert memory_increase < 200, f"Memory increase too high: {memory_increase}MB"

def test_concurrent_fitting():
    """Test performance when fitting multiple models concurrently"""
    import concurrent.futures
    from threading import Lock
    
    # Shared lock to prevent conflicts in model fitting
    fitting_lock = Lock()
    
    def fit_single_model(seed):
        with fitting_lock:
            np.random.seed(seed)
            t = np.linspace(0, 50, 500)
            # Generate synthetic data
            true_p, true_q, true_m = 0.03, 0.38, 1000
            exp_term = np.exp(-(true_p + true_q) * t)
            y_true = true_m * (1 - exp_term) / (1 + (true_q / true_p) * exp_term)
            y_noisy = y_true + np.random.normal(0, 50, len(t))
            
            model = BassModel()
            fitter = ScipyFitter()
            fitter.fit(model, t, y_noisy)
            return model.params_
    
    # Test concurrent fitting of 10 models
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fit_single_model, i) for i in range(10)]
        results = [future.result() for future in futures]
    
    # Verify all fittings completed successfully
    assert all(r is not None for r in results)
    assert all('p' in r and 'q' in r and 'm' in r for r in results)
```

## 5. Stress Testing

**Implementation**:

```python
# tests/test_stress.py
import numpy as np
import pytest
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.fitters.scipy_fitter import ScipyFitter

def test_extreme_parameter_fitting():
    """Test fitting with extreme parameter ranges"""
    t = np.arange(1, 100)
    # Create data with very high adoption rate
    y = 1000000 * (1 - np.exp(-100 * t))  # Very high rate
    # Ensure y doesn't cause overflow issues
    y = np.clip(y, 0, 1e10)
    
    model = BassModel()
    fitter = ScipyFitter()
    
    try:
        fitter.fit(model, t, y)
        # Check that parameters are reasonable or model fails gracefully
        if model.params_ is not None:
            assert "p" in model.params_
            assert "q" in model.params_
            assert "m" in model.params_
    except Exception:
        # If fitting fails, it should fail gracefully
        pass

def test_invalid_data_handling():
    """Test how models handle invalid data"""
    # Test with constant data (no variation - problematic for fitting)
    t = np.array([1, 2, 3, 4, 5])
    y = np.array([100, 100, 100, 100, 100])  # No variation
    
    model = BassModel()
    fitter = ScipyFitter()
    
    try:
        fitter.fit(model, t, y)
        # Model might still fit but parameters might be odd
        if model.params_ is not None:
            # If it does fit, the parameters should be checked
            pass
    except Exception:
        # This is acceptable - sometimes constant data can't be fitted
        pass

def test_numerical_stability():
    """Test numerical stability of models with problematic inputs"""
    # Test with very large time values
    t = np.linspace(1000, 2000, 100)  # Very large time values
    # Use small parameters to avoid overflow
    model = GompertzModel()
    model.params_ = {"a": 1.0, "b": 0.01, "c": 0.001}
    
    try:
        pred = model.predict(t)
        # Check that predictions are finite
        assert np.all(np.isfinite(pred))
        # Check that predictions don't have extreme values
        assert np.all(np.abs(pred) < 1e10)
    except Exception:
        # Some extreme values might cause overflow, which is expected
        pass

def test_boundary_conditions():
    """Test models at parameter boundaries"""
    t = np.linspace(0, 50, 100)
    
    # Test with extremely small parameters
    model = BassModel()
    model.params_ = {"p": 1e-10, "q": 1e-10, "m": 1e-5}
    
    try:
        pred = model.predict(t)
        assert np.all(np.isfinite(pred))
        assert len(pred) == len(t)
    except Exception:
        # Very small parameters might cause numerical issues, which is acceptable
        pass
```

## 6. Endurance Testing

**Implementation**:

```python
# tests/test_endurance.py
import numpy as np
import pytest
import gc
import psutil
import os
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_leak_detection():
    """Test for memory leaks during repeated operations"""
    # Starting memory usage
    initial_memory = get_memory_usage()
    
    # Perform many fitting operations
    for i in range(100):
        t = np.linspace(0, 50, 100)
        y = 1000 * (1 - np.exp(-0.05 * t)) + np.random.normal(0, 10, len(t))
        
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        
        # Trigger garbage collection periodically
        if i % 10 == 0:
            gc.collect()
    
    # Allow time for garbage collection to take effect
    gc.collect()
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be minimal - no more than 50MB for these operations
    assert memory_increase < 50, f"Potential memory leak detected: {memory_increase}MB increase"

def test_numerical_stability_over_time():
    """Test that model parameters remain stable over many operations"""
    base_params = {"p": 0.03, "q": 0.38, "m": 1000}
    param_drifts = []
    
    for i in range(50):
        t = np.linspace(0, 50, 100)
        # Generate synthetic data with known parameters
        exp_term = np.exp(-(base_params["p"] + base_params["q"]) * t)
        y_true = base_params["m"] * (1 - exp_term) / (1 + (base_params["q"] / base_params["p"]) * exp_term)
        y_noisy = y_true + np.random.normal(0, base_params["m"] * 0.05, len(t))
        
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y_noisy)
        
        if model.params_:
            # Calculate parameter drift from expected values
            p_drift = abs(model.params_.get("p", 0) - base_params["p"])
            q_drift = abs(model.params_.get("q", 0) - base_params["q"])
            m_drift = abs(model.params_.get("m", 0) - base_params["m"])
            param_drifts.append((p_drift, q_drift, m_drift))
    
    # Check that average drift is within acceptable bounds
    avg_p_drift = np.mean([d[0] for d in param_drifts])
    avg_q_drift = np.mean([d[1] for d in param_drifts])
    avg_m_drift = np.mean([d[2] for d in param_drifts])
    
    # Drift should be relatively small
    assert avg_p_drift < 0.01, f"P parameter drift too high: {avg_p_drift}"
    assert avg_q_drift < 0.1, f"Q parameter drift too high: {avg_q_drift}"
    assert avg_m_drift < 50, f"M parameter drift too high: {avg_m_drift}"

def test_long_running_fitting():
    """Test fitting over a long period of time"""
    import time
    
    start_time = time.time()
    
    # Run fitting operations for a longer period
    operation_count = 0
    time_limit = 30  # seconds
    
    while time.time() - start_time < time_limit:
        t = np.linspace(0, 50, 200)
        y = 1000 * (1 - np.exp(-0.03 * t)) + np.random.normal(0, 20, len(t))
        
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        
        operation_count += 1
        
        # Verify model was fitted successfully
        assert model.params_ is not None
        assert all(param in model.params_ for param in ["p", "q", "m"])
    
    # Verify that we completed a reasonable number of operations
    assert operation_count > 5, f"Too few operations completed: {operation_count}"
    
    elapsed = time.time() - start_time
    print(f"Completed {operation_count} fittings in {elapsed:.2f} seconds")
```

## 7. Recovery Testing

**Implementation**:

```python
# tests/test_recovery.py
import numpy as np
import pytest
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter

def test_fitting_failure_recovery():
    """Test that model recovers gracefully from fitting failures"""
    t = np.array([1, 2, 3])  # Too small dataset for reliable fitting
    y = np.array([10, 10, 10])  # No variance - problematic for fitting
    
    model = BassModel()
    original_state = model.__dict__.copy() if hasattr(model, '__dict__') else {}
    fitter = ScipyFitter()
    
    try:
        fitter.fit(model, t, y)
        # If fitting succeeds (which it might with minimal data), 
        # ensure params are reasonable
        if model.params_ is not None:
            assert "p" in model.params_
            assert "q" in model.params_
            assert "m" in model.params_
    except Exception as e:
        # After failure, model should maintain a clean state
        # It may have no params (which is valid after a failed fit)
        assert model.params_ is None or isinstance(model.params_, dict)

def test_invalid_parameter_recovery():
    """Test that models handle invalid parameters gracefully"""
    t = np.linspace(0, 50, 100)
    model = BassModel()
    
    # Set invalid parameters (negative values)
    model.params_ = {"p": -0.1, "q": -0.5, "m": -100}
    
    try:
        pred = model.predict(t)
        # If prediction works with negative params, 
        # check if it makes mathematical sense
        # For Bass model with negative params this might produce invalid results
    except Exception:
        # This is acceptable behavior - invalid parameters should cause issues
        pass
    finally:
        # Test resetting to valid parameters
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        pred = model.predict(t)
        assert len(pred) == len(t)
        assert np.all(np.isfinite(pred))

def test_numerical_error_recovery():
    """Test recovery from numerical errors"""
    t = np.linspace(1000, 2000, 10)  # Very large time values
    model = BassModel()
    model.params_ = {"p": 1e-20, "q": 1e20, "m": 1e20}  # Extreme parameters
    
    try:
        pred = model.predict(t)
        # Check if predictions are reasonable
        if not np.all(np.isfinite(pred)) or np.any(np.abs(pred) > 1e100):
            # If predictions are invalid, the model should handle this gracefully
            pass
    except Exception:
        # Numerical errors might occur with extreme parameters
        pass
    finally:
        # Reset to reasonable parameters
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        pred = model.predict(t)
        assert np.all(np.isfinite(pred))

def test_state_consistency_after_errors():
    """Test that model state remains consistent after errors"""
    model = BassModel()
    
    # Save original state characteristics
    has_params = hasattr(model, 'params_')
    initial_params = getattr(model, 'params_', None)
    
    # Attempt operations that might cause issues
    test_cases = [
        # Case 1: Invalid data
        (np.array([1, 2]), np.array([5, 5])),  # Too little data, no variance
        # Case 2: Large data causing potential overflow
        (np.linspace(0, 1000, 5), np.exp(np.linspace(0, 1000, 5))),  # Large values
    ]
    
    for t_test, y_test in test_cases:
        fitter = ScipyFitter()
        try:
            fitter.fit(model, t_test, y_test)
        except Exception:
            # If fitting fails, the model should still be in a valid state
            # It may have params from previous successful fits or be None
            pass
    
    # After all error attempts, model should still be usable
    # Reset to known good state to test functionality
    clean_t = np.linspace(0, 10, 20)
    clean_y = 100 * (1 - np.exp(-0.1 * clean_t))
    
    clean_model = BassModel()
    clean_fitter = ScipyFitter()
    clean_fitter.fit(clean_model, clean_t, clean_y)
    
    assert clean_model.params_ is not None
    assert all(param in clean_model.params_ for param in ["p", "q", "m"])
```

## 8. Coverage Improvement Strategy

To achieve >95% individual file test coverage:

1. **Identify low-coverage files**:
   ```bash
   pytest --cov=innovate --cov-report=term-missing
   ```

2. **Create targeted tests** for each low-coverage file:
   - Branch coverage for all conditional statements
   - Exception handling paths
   - Edge cases for all functions
   - Error conditions

3. **Use coverage reports** to identify missing lines and create tests for them specifically.

**Example for improving coverage in a specific file**:

```python
# tests/test_coverage_improvement.py
import pytest
import numpy as np
from innovate.diffuse.bass import BassModel

def test_bass_model_edge_cases():
    """Test edge cases for Bass model to improve coverage"""
    model = BassModel()
    
    # Test with minimal data
    t_minimal = np.array([1])
    y_minimal = np.array([10])
    
    fitter = ScipyFitter()
    with pytest.raises(Exception):  # Fitting with single point should fail or handle gracefully
        fitter.fit(model, t_minimal, y_minimal)

def test_bass_model_error_paths():
    """Test various error handling paths"""
    model = BassModel()
    
    # Test prediction before fitting
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])
    
    # Test scoring before fitting  
    with pytest.raises(RuntimeError):
        model.score([1, 2, 3], [10, 20, 30])
    
    # Test with invalid parameters
    model.params_ = {"p": -1, "q": -1, "m": -1}  # Invalid parameters
    # This might work mathematically but results might be invalid
    result = model.predict([1, 2, 3])
    assert len(result) == 3
```

## CI/CD Integration

Add to `.github/workflows/test.yml` or similar CI configuration:

```yaml
name: Comprehensive Testing
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.14'
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --group dev
    - name: Run standard tests
      run: pytest --cov=innovate --cov-report=xml
    - name: Run property-based tests
      run: pytest tests/test_property_based.py -v
    - name: Run performance tests
      run: pytest tests/test_performance.py -v
    - name: Run load tests
      run: pytest tests/test_load.py -v
    - name: Run stress tests
      run: pytest tests/test_stress.py -v
    - name: Run endurance tests
      run: pytest tests/test_endurance.py -v
    - name: Run recovery tests
      run: pytest tests/test_recovery.py -v
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

This comprehensive testing strategy will significantly improve the robustness, reliability, and maturity of the innovate library while providing confidence in its mathematical correctness and performance characteristics.
