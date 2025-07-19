import timeit
import pandas as pd
import numpy as np
from innovate.diffuse import BassModel, GompertzModel, LogisticModel
from innovate.fitters import ScipyFitter
from innovate.backend import use_backend

def run_fit_benchmark(model, t, y, backend, fitter):
    """Runs a fit benchmark for a given model, backend, and fitter."""
    use_backend(backend)

    # Time the fitting process
    fit_time = timeit.timeit(lambda: fitter.fit(model, t, y), number=10)

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": fitter.__class__.__name__,
        "task": "fit",
        "time": fit_time,
    }

def run_predict_benchmark(model, t, backend):
    """Runs a predict benchmark for a given model and backend."""
    use_backend(backend)

    # Time the prediction process
    predict_time = timeit.timeit(lambda: model.predict(t), number=100)

    return {
        "model": model.__class__.__name__,
        "backend": backend,
        "fitter": None,
        "task": "predict",
        "time": predict_time,
    }

def run_simulation_benchmark(model, t, backend, n_sims):
    """Runs a simulation benchmark for a given model and backend."""
    use_backend(backend)

    # Time the simulation process
    sim_time = timeit.timeit(lambda: [model.predict(t) for _ in range(n_sims)], number=1)

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

    # --- Bass Model ---
    true_bass = BassModel()
    true_bass.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
    y_bass = true_bass.predict(t) + np.random.normal(0, 10, len(t)) # add some noise
    bass_model = BassModel() # This is the model instance to be used in benchmarks

    # --- Gompertz Model ---
    true_gompertz = GompertzModel()
    true_gompertz.params_ = {"a": 1000, "b": 5, "c": 0.1}
    y_gompertz = true_gompertz.predict(t) + np.random.normal(0, 10, len(t))
    gompertz_model = GompertzModel()

    # --- Logistic Model ---
    true_logistic = LogisticModel()
    true_logistic.params_ = {"L": 1000, "k": 0.1, "x0": 25}
    y_logistic = true_logistic.predict(t) + np.random.normal(0, 10, len(t))
    logistic_model = LogisticModel()

    # Complex diffusion model with covariates
    covariates = {"price": np.linspace(10, 5, 100)}
    bass_model_cov = BassModel(covariates=list(covariates.keys()))
    true_bass_cov = BassModel(covariates=list(covariates.keys()))
    true_bass_cov.params_ = {"p": 0.03, "q": 0.38, "m": 1000, "beta_p_price": 0.01, "beta_q_price": -0.02, "beta_m_price": 10}
    y_bass_cov = true_bass_cov.predict(t, covariates=covariates) + np.random.normal(0, 10, len(t))


    # Create the fitters
    scipy_fitter = ScipyFitter()

    # Run the benchmarks
    results = []
    backends = ["numpy", "jax"]

    models_to_benchmark = [
        (bass_model, y_bass, None, "BassModel"),
        (gompertz_model, y_gompertz, None, "GompertzModel"),
        (logistic_model, y_logistic, None, "LogisticModel"),
        (bass_model_cov, y_bass_cov, covariates, "BassModel (Cov)"),
    ]

    # Fit benchmarks
    for model, y, covs, _ in models_to_benchmark:
        for backend in backends:
            # Need to fit the model first before benchmarking prediction/simulation
            fitter = ScipyFitter()
            if covs:
                fitter.fit(model, t, y, covariates=covs)
            else:
                fitter.fit(model, t, y)
            results.append(run_fit_benchmark(model, t, y, backend, scipy_fitter))

    # Predict and Simulate benchmarks
    for model, _, covs, _ in models_to_benchmark:
        for backend in backends:
            if covs:
                results.append(run_predict_benchmark(model, t, backend, covariates=covs))
                for n_sims in [10, 100, 1000]:
                    results.append(run_simulation_benchmark(model, t, backend, n_sims, covariates=covs))
            else:
                results.append(run_predict_benchmark(model, t, backend))
                for n_sims in [10, 100, 1000]:
                    results.append(run_simulation_benchmark(model, t, backend, n_sims))

    # Print the results
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()