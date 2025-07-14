import timeit
import pandas as pd
import numpy as np
from innovate.diffuse import BassModel, GompertzModel, LogisticModel
from innovate.fitters import ScipyFitter
from innovate.backend import use_backend

def run_fit_benchmark(model, t, y, backend, fitter, covariates=None):
    """Runs a fit benchmark for a given model, backend, and fitter."""
    use_backend(backend)

    # Time the fitting process
    fit_time = timeit.timeit(lambda: fitter.fit(model, t, y, covariates=covariates), number=10)

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": fitter.__class__.__name__,
        "task": "fit",
        "time": fit_time,
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

def main():
    """Runs the benchmarks and prints the results."""

    # Create some synthetic data
    t = np.linspace(0, 50, 100)

    # Simple diffusion models
    bass_model = BassModel()
    gompertz_model = GompertzModel()
    logistic_model = LogisticModel()

    fitter = ScipyFitter()
    fitter.fit(bass_model, t, np.sin(t))
    fitter.fit(gompertz_model, t, np.sin(t))
    fitter.fit(logistic_model, t, np.sin(t))

    y_bass = bass_model.predict(t)
    y_gompertz = gompertz_model.predict(t)
    y_logistic = logistic_model.predict(t)

    # Complex diffusion model with covariates
    covariates = {"price": np.linspace(10, 5, 100)}
    bass_model_cov = BassModel(covariates=list(covariates.keys()))
    fitter.fit(bass_model_cov, t, np.sin(t), covariates=covariates)
    y_bass_cov = bass_model_cov.predict(t, covariates=covariates)

    # Create the fitters
    scipy_fitter = ScipyFitter()

    # Run the benchmarks
    results = []

    # Fit benchmarks
    for model, y in [(bass_model, y_bass), (gompertz_model, y_gompertz), (logistic_model, y_logistic)]:
        for backend in ["numpy", "jax"]:
            results.append(run_fit_benchmark(model, t, y, backend, scipy_fitter))

    results.append(run_fit_benchmark(bass_model_cov, t, y_bass_cov, "numpy", scipy_fitter, covariates=covariates))
    results.append(run_fit_benchmark(bass_model_cov, t, y_bass_cov, "jax", scipy_fitter, covariates=covariates))

    # Predict benchmarks
    for model in [bass_model, gompertz_model, logistic_model]:
        for backend in ["numpy", "jax"]:
            results.append(run_predict_benchmark(model, t, backend))

    results.append(run_predict_benchmark(bass_model_cov, t, "numpy", covariates=covariates))
    results.append(run_predict_benchmark(bass_model_cov, t, "jax", covariates=covariates))

    # Simulation benchmarks
    for model in [bass_model, gompertz_model, logistic_model]:
        for backend in ["numpy", "jax"]:
            for n_sims in [10, 100, 1000]:
                results.append(run_simulation_benchmark(model, t, backend, n_sims))

    for backend in ["numpy", "jax"]:
        for n_sims in [10, 100, 1000]:
            results.append(run_simulation_benchmark(bass_model_cov, t, backend, n_sims, covariates=covariates))

    # Print the results
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()
