"""Benchmarks for the innovate library."""
import timeit
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from innovate.backend import use_backend
from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import ScipyFitter


def run_fit_benchmark(
    model: Any,
    t: np.ndarray,
    y: np.ndarray,
    backend: str,
    fitter: Any,
    covariates: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run a fit benchmark for a given model, backend, and fitter."""
    try:
        use_backend(backend)
    except ValueError as e:
        return {
            "model": model.__class__.__name__,
            "backend": backend,
            "fitter": fitter.__class__.__name__,
            "task": "fit",
            "time": None,
            "error": str(e),
        }

    # Time the fitting process
    try:
        fit_time = timeit.timeit(
            lambda: fitter.fit(model, t, y, covariates=covariates),
            number=10,
        )
    except ValueError as e:
        fit_time = None
        error = str(e)
    else:
        error = None

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": fitter.__class__.__name__,
        "task": "fit",
        "time": fit_time,
        "error": error,
    }


def run_predict_benchmark(
    model: Any,
    t: np.ndarray,
    backend: str,
    covariates: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run a predict benchmark for a given model and backend."""
    use_backend(backend)

    # Time the prediction process
    predict_time = timeit.timeit(
        lambda: model.predict(t, covariates=covariates),
        number=100,
    )

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": None,
        "task": "predict",
        "time": predict_time,
    }


def run_simulation_benchmark(
    model: Any,
    t: np.ndarray,
    backend: str,
    n_sims: int,
    covariates: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Run a simulation benchmark for a given model and backend."""
    use_backend(backend)

    # Time the simulation process
    sim_time = timeit.timeit(
        lambda: [model.predict(t, covariates=covariates) for _ in range(n_sims)],
        number=1,
    )

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": None,
        "task": f"simulate_{n_sims}",
        "time": sim_time,
    }


def generate_synthetic_data(
    model: Any,
    t: np.ndarray,
    params: Dict[str, float],
    covariates: Optional[Dict[str, np.ndarray]] = None,
    noise_std: float = 0.05,
) -> np.ndarray:
    """Generate synthetic data from a model with known parameters and adds noise."""
    model.params_ = params
    y_true = model.predict(t, covariates=covariates)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_std * np.max(y_true), len(t))
    y_noisy = y_true + noise
    return np.maximum(0, y_noisy)  # ensure non-negative


def main() -> None:
    """Run the benchmarks and prints the results."""
    t = np.linspace(0, 50, 100)
    results: List[Dict[str, Any]] = []
    backends = ["numpy", "jax"]
    fitter = ScipyFitter()

    # --- Model Configurations ---
    models_to_benchmark: List[
        Tuple[Any, Dict[str, float], Optional[Dict[str, np.ndarray]], str]
    ] = [
        (BassModel(), {"p": 0.03, "q": 0.38, "m": 1000}, None, "Bass"),
        (GompertzModel(), {"a": 1000, "b": 5, "c": 0.1}, None, "Gompertz"),
        (LogisticModel(), {"L": 1000, "k": 0.1, "x0": 25}, None, "Logistic"),
    ]

    # Add model with covariates
    covariates = {"price": np.linspace(10, 5, 100)}
    bass_model_cov = BassModel(covariates=list(covariates.keys()))
    bass_cov_params = {
        "p": 0.03,
        "q": 0.38,
        "m": 1000,
        "beta_p_price": -0.001,
        "beta_q_price": 0.01,
        "beta_m_price": 10,
    }
    models_to_benchmark.append(
        (bass_model_cov, bass_cov_params, covariates, "Bass (Covariates)"),
    )

    for model, params, covs, name in models_to_benchmark:
        y = generate_synthetic_data(model, t, params, covariates=covs)

        for backend in backends:
            try:
                use_backend(backend)
                # Fit the model first before benchmarking predict/simulate
                fitter.fit(model, t, y, covariates=covs)

                # Run benchmarks
                results.append(
                    run_fit_benchmark(model, t, y, backend, fitter, covariates=covs),
                )
                results.append(
                    run_predict_benchmark(model, t, backend, covariates=covs),
                )
                for n_sims in [10, 100, 1000]:
                    results.append(
                        run_simulation_benchmark(
                            model,
                            t,
                            backend,
                            n_sims,
                            covariates=covs,
                        ),
                    )
            except ValueError as e:
                results.append(
                    {
                        "model": name,
                        "backend": backend,
                        "fitter": fitter.__class__.__name__,
                        "task": "fit",
                        "time": None,
                        "error": str(e),
                    },
                )

    # Print the results
    results_df = pd.DataFrame(results)
    with open("benchmark_results.txt", "w") as f:
        f.write("\n--- Benchmark Results ---\n")
        f.write(results_df.to_string())
    results_df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
