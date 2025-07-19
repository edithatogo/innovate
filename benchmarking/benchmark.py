import timeit
import pandas as pd
import numpy as np
from innovate.diffuse import BassModel, GompertzModel, LogisticModel
from innovate.fitters import ScipyFitter
from innovate.backend import use_backend

def run_fit_benchmark(model, t, y, backend, fitter, covariates=None):
    """Runs a fit benchmark for a given model, backend, and fitter."""
    try:
        use_backend(backend)
    except Exception as e:
        return {
            "model": model.__class__.__name__,
            "backend": backend,
            "fitter": fitter.__class__.__name__,
            "task": "fit",
            "time": None,
            "error": str(e)
        }

    # Time the fitting process
    try:
        fit_time = timeit.timeit(lambda: fitter.fit(model, t, y, covariates=covariates), number=10)
    except Exception as e:
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
        "error": error
    }

def run_predict_benchmark(model, t, backend, covariates=None):
    """Runs a predict benchmark for a given model and backend."""
    use_backend(backend)

    # Time the prediction process
    predict_time = timeit.timeit(lambda: model.predict(t, covariates=covariates), number=100)

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": None,
        "task": "predict",
        "time": predict_time,
    }

def run_simulation_benchmark(model, t, backend, n_sims, covariates=None):
    """Runs a simulation benchmark for a given model and backend."""
    use_backend(backend)

    # Time the simulation process
    sim_time = timeit.timeit(lambda: [model.predict(t, covariates=covariates) for _ in range(n_sims)], number=1)

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": None,
        "task": f"simulate_{n_sims}",
        "time": sim_time,
    }

def generate_synthetic_data(model, t, params, covariates=None, noise_std=10):
    """Generates synthetic data from a model with known parameters and adds noise."""
    model.params_ = params
    y = model.predict(t, covariates=covariates) + np.random.normal(0, noise_std, len(t))
    return y

def main():
    """Runs the benchmarks and prints the results."""

    # Create some synthetic data
    t = np.linspace(0, 50, 100)

    # Define models and their corresponding data/covariates
    models_to_benchmark = [
        (
            BassModel(),
            {"p": 0.03, "q": 0.38, "m": 1000},
            None,
            "BassModel"
        ),
        (
            GompertzModel(),
            {"a": 1000, "b": 5, "c": 0.1},
            None,
            "GompertzModel"
        ),
        (
            LogisticModel(),
            {"L": 1000, "k": 0.1, "x0": 25},
            None,
            "LogisticModel"
        ),
        (
            BassModel(covariates=["price"]),
            {"p": 0.03, "q": 0.38, "m": 1000, "beta_p_price": 0.01, "beta_q_price": -0.02, "beta_m_price": 10},
            {"price": np.linspace(10, 5, 100)},
            "BassModel (Cov)"
        ),
    ]

    # Create the fitters
    scipy_fitter = ScipyFitter()

    # Run the benchmarks
    results = []
    backends = ["numpy", "jax"]

    for model, params, covs, name in models_to_benchmark:
        y = generate_synthetic_data(model, t, params, covariates=covs)
        for backend in backends:
            # Fit the model first before benchmarking
            fitter = ScipyFitter()
            fitter.fit(model, t, y, covariates=covs)
            results.append(run_fit_benchmark(model, t, y, backend, fitter, covariates=covs))
            results.append(run_predict_benchmark(model, t, backend, covariates=covs))
            for n_sims in [10, 100, 1000]:
                results.append(run_simulation_benchmark(model, t, backend, n_sims, covariates=covs))

    # Print the results
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()