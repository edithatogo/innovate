import timeit
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from heartflow.models.logistic import LogisticModel
from heartflow.fitters.scipy_fitter import ScipyFitter
from heartflow.fitters.jax_fitter import JaxFitter
from heartflow.fitters.batched_fitter import BatchedFitter
from heartflow.backend import use_backend

def generate_data(n_samples, n_datasets):
    t = np.linspace(0, 20, n_samples)
    
    t_batched = [t] * n_datasets
    y_batched = []
    for i in range(n_datasets):
        L = 1.0 + i * 0.1
        k = 1.5 + i * 0.05
        x0 = 10.0 + i * 0.2
        y = L / (1 + np.exp(-k * (t - x0))) + np.random.normal(0, 0.01, len(t))
        y_batched.append(y)
        
    return t_batched, y_batched

def run_benchmarks():
    n_samples = 100
    n_datasets_single = 1
    n_datasets_batched = 10

    t_single, y_single = generate_data(n_samples, n_datasets_single)
    t_batched, y_batched = generate_data(n_samples, n_datasets_batched)

    model = LogisticModel()

    # --- Single Fit Benchmarks ---
    print("--- Single Fit Benchmarks ---")
    
    # SciPy Fitter
    use_backend("numpy")
    scipy_fitter = ScipyFitter()
    scipy_time = timeit.timeit(lambda: scipy_fitter.fit(model, t_single[0], y_single[0]), number=10)
    print(f"SciPyFitter time: {scipy_time:.4f}s")

    # JAX Fitter
    use_backend("jax")
    jax_fitter = JaxFitter()
    jax_time = timeit.timeit(lambda: jax_fitter.fit(model, t_single[0], y_single[0]), number=10)
    print(f"JAXFitter time: {jax_time:.4f}s")


    # --- Batched Fit Benchmarks ---
    print("\n--- Batched Fit Benchmarks ---")
    
    # Batched Fitter with NumPy backend
    use_backend("numpy")
    batched_fitter_numpy = BatchedFitter(model, ScipyFitter())
    numpy_batch_time = timeit.timeit(lambda: batched_fitter_numpy.fit(t_batched, y_batched), number=3)
    print(f"BatchedFitter (NumPy) time: {numpy_batch_time:.4f}s")

    # Batched Fitter with JAX backend
    use_backend("jax")
    batched_fitter_jax = BatchedFitter(model, JaxFitter())
    jax_batch_time = timeit.timeit(lambda: batched_fitter_jax.fit(t_batched, y_batched), number=3)
    print(f"BatchedFitter (JAX) time: {jax_batch_time:.4f}s")

if __name__ == "__main__":
    run_benchmarks()
