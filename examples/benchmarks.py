"""Benchmarks for the innovate library."""
import sys
import timeit
from pathlib import Path
from typing import List, Tuple

import numpy as np
from innovate.backend import use_backend
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.batched_fitter import BatchedFitter
from innovate.fitters.jax_fitter import JaxFitter
from innovate.fitters.scipy_fitter import ScipyFitter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_data(
    n_samples: int,
    n_datasets: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate synthetic data for benchmarking."""
    t = np.linspace(0, 20, n_samples)

    t_batched = [t] * n_datasets
    y_batched = []
    rng = np.random.default_rng(42)
    for i in range(n_datasets):
        limit = 1.0 + i * 0.1
        k = 1.5 + i * 0.05
        x0 = 10.0 + i * 0.2
        y = limit / (1 + np.exp(-k * (t - x0))) + rng.normal(0, 0.01, len(t))
        y_batched.append(y)

    return t_batched, y_batched


def run_benchmarks() -> None:
    """Run the benchmarks."""
    n_samples = 100
    n_datasets_single = 1
    n_datasets_batched = 10

    t_single, y_single = generate_data(n_samples, n_datasets_single)
    t_batched, y_batched = generate_data(n_samples, n_datasets_batched)

    model = LogisticModel()

    # --- Single Fit Benchmarks ---
    # SciPy Fitter
    use_backend("numpy")
    scipy_fitter = ScipyFitter()
    timeit.timeit(
        lambda: scipy_fitter.fit(model, t_single[0], y_single[0]),
        number=10,
    )

    # JAX Fitter
    use_backend("jax")
    jax_fitter = JaxFitter()
    timeit.timeit(
        lambda: jax_fitter.fit(model, t_single[0], y_single[0]),
        number=10,
    )

    # --- Batched Fit Benchmarks ---
    # Batched Fitter with NumPy backend
    use_backend("numpy")
    batched_fitter_numpy = BatchedFitter(model, ScipyFitter())
    timeit.timeit(
        lambda: batched_fitter_numpy.fit(t_batched, y_batched),
        number=3,
    )

    # Batched Fitter with JAX backend
    use_backend("jax")
    batched_fitter_jax = BatchedFitter(model, JaxFitter())
    timeit.timeit(
        lambda: batched_fitter_jax.fit(t_batched, y_batched),
        number=3,
    )


if __name__ == "__main__":
    run_benchmarks()
