"""Usage examples for innovate diffusion modeling library.

This module provides runnable examples demonstrating key features:
1. Basic diffusion curve fitting
2. Model comparison and selection
3. Enhanced fitting with multiple optimization methods
4. Fit diagnostics and residual analysis
5. Bootstrap confidence intervals
6. Mixture models for segment identification
"""

import numpy as np


def example_basic_fitting():
    """Example 1: Basic diffusion curve fitting with Bass model.

    Demonstrates the simplest workflow: generate synthetic data,
    fit a Bass model, and evaluate the fit.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.scipy_fitter import ScipyFitter

    # Generate synthetic Bass model data
    np.random.seed(42)
    t = np.linspace(0, 12, 40)
    p_true, q_true, m_true = 0.03, 0.38, 100.0
    y_true = m_true * (1 - np.exp(-(p_true + q_true) * t)) / (1 + (q_true / p_true) * np.exp(-(p_true + q_true) * t))
    y_noisy = y_true + np.random.normal(0, 2, len(t))
    y_noisy = np.maximum(y_noisy, 0)  # Ensure non-negative

    # Fit the model
    model = BassModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, y_noisy)

    print("Fitted parameters:")
    for name, value in model.params_.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nR² score: {model.score(t, y_noisy):.4f}")

    # Predict on new time points
    t_new = np.linspace(0, 15, 50)
    y_pred = model.predict(t_new)
    print(f"Prediction at t=15: {y_pred[-1]:.2f}")


def example_model_comparison():
    """Example 2: Model comparison and selection.

    Demonstrates fitting multiple diffusion models to the same data
    and selecting the best based on goodness-of-fit metrics.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.diffuse.gompertz import GompertzModel
    from innovate.diffuse.logistic import LogisticModel
    from innovate.fitters.scipy_fitter import ScipyFitter

    # Generate synthetic data from Bass model
    np.random.seed(42)
    t = np.linspace(0, 12, 40)
    p, q, m = 0.03, 0.38, 100.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 2, len(t))
    y = np.maximum(y, 0)

    # Define models to compare
    models = {
        "Bass": BassModel(),
        "Gompertz": GompertzModel(),
        "Logistic": LogisticModel(),
    }

    # Fit each model and collect diagnostics
    fitter = ScipyFitter(store_diagnostics=True)
    results = {}

    for name, model in models.items():
        try:
            fitter.fit(model, t, y)
            results[name] = {
                "model": model,
                "r_squared": fitter.diagnostics.r_squared,
                "rmse": fitter.diagnostics.rmse,
                "aic": fitter.diagnostics.aic,
                "bic": fitter.diagnostics.bic,
            }
            print(f"{name}: R²={fitter.diagnostics.r_squared:.4f}, AIC={fitter.diagnostics.aic:.2f}")
        except RuntimeError as e:  # noqa: PERF203
            print(f"{name}: Fitting failed - {e}")

    # Select best model by AIC
    best_model = min(results.items(), key=lambda x: x[1]["aic"])
    print(f"\nBest model by AIC: {best_model[0]}")


def example_optimization_methods():
    """Example 3: Comparing different optimization methods.

    Demonstrates the various optimization methods available in ScipyFitter
    and when to use each one.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.scipy_fitter import ScipyFitter

    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 12, 40)
    p, q, m = 0.03, 0.38, 100.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 2, len(t))
    y = np.maximum(y, 0)

    methods = ["curve_fit", "least_squares", "nelder_mead", "lbfgsb", "differential_evolution"]

    print("Comparing optimization methods:")
    print(f"{'Method':<25} {'R²':>10} {'RMSE':>10} {'Converged':>10}")
    print("-" * 55)

    for method in methods:
        model = BassModel()
        fitter = ScipyFitter(method=method, maxiter=500, store_diagnostics=True)
        try:
            fitter.fit(model, t, y)
            diag = fitter.diagnostics
            print(f"{method:<25} {diag.r_squared:>10.4f} {diag.rmse:>10.4f} {diag.convergence_status:>10}")
        except Exception as e:
            print(f"{method:<25} {'FAILED':>10} {'':>10} {'':>10} - {str(e)[:30]}")

    # Automatic method selection
    model_auto = BassModel()
    fitter_auto = ScipyFitter(method="auto", store_diagnostics=True)
    fitter_auto.fit(model_auto, t, y)
    print(f"\nAuto-selected method: {fitter_auto.diagnostics.optimization_method}")


def example_fit_diagnostics():
    """Example 4: Fit diagnostics and residual analysis.

    Demonstrates comprehensive goodness-of-fit metrics and
    residual analysis for model validation.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.residual_analysis import analyze_residuals
    from innovate.fitters.scipy_fitter import ScipyFitter

    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 12, 40)
    p, q, m = 0.03, 0.38, 100.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 2, len(t))
    y = np.maximum(y, 0)

    # Fit with diagnostics
    model = BassModel()
    fitter = ScipyFitter(store_diagnostics=True)
    fitter.fit(model, t, y)

    # Print diagnostics summary
    print(fitter.diagnostics.summary())

    # Perform residual analysis
    y_pred = model.predict(t)
    residuals = np.array(y) - np.array(y_pred)
    analysis = analyze_residuals(residuals, np.array(y_pred))

    print("\n" + analysis.summary())

    # Interpret results
    print("\nDiagnostic interpretation:")
    print(f"  Residuals normally distributed: {analysis.is_normally_distributed()}")
    print(f"  Residuals show autocorrelation: {analysis.has_autocorrelation()}")
    if analysis.breusch_pagan_p is not None:
        print(f"  Heteroscedasticity detected: {analysis.has_heteroscedasticity()}")


def example_bootstrap_confidence_intervals():
    """Example 5: Bootstrap confidence intervals for parameter uncertainty.

    Demonstrates estimating parameter uncertainty using bootstrapping,
    which is essential for understanding the reliability of fitted parameters.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.bootstrap_fitter import BootstrapFitter
    from innovate.fitters.scipy_fitter import ScipyFitter

    # Generate synthetic data
    np.random.seed(42)
    t = np.linspace(0, 12, 40)
    p, q, m = 0.03, 0.38, 100.0
    y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    y += np.random.normal(0, 2, len(t))
    y = np.maximum(y, 0)

    # Fit base model
    model = BassModel()
    base_fitter = ScipyFitter()
    base_fitter.fit(model, t, y)

    print("Point estimates:")
    for name, value in model.params_.items():
        print(f"  {name}: {value:.4f}")

    # Bootstrap confidence intervals
    print("\nRunning bootstrap (this may take a moment)...")
    bootstrap_fitter = BootstrapFitter(base_fitter, n_bootstraps=50, seed=42)
    bootstrap_fitter.fit(model, t, y)

    # Get confidence intervals
    cis = bootstrap_fitter.get_confidence_intervals(alpha=0.05)
    print("\n95% Confidence Intervals:")
    print(f"{'Parameter':<15} {'Lower':>10} {'Median':>10} {'Upper':>10}")
    print("-" * 45)
    for name, ci in cis.items():
        print(f"{name:<15} {ci['lower']:>10.4f} {ci['median']:>10.4f} {ci['upper']:>10.4f}")

    # Print full summary
    print("\n" + bootstrap_fitter.summary())

    # Parameter correlation
    corr = bootstrap_fitter.get_parameter_correlation()
    if corr:
        print("\nParameter Correlations:")
        params = list(corr.keys())
        header = f"{'':>15}" + "".join(f"{p:>10}" for p in params)
        print(header)
        for p1 in params:
            row = f"{p1:>15}" + "".join(f"{corr[p1][p2]:>10.3f}" for p2 in params)
            print(row)


def example_mixture_model():
    """Example 6: Mixture models for identifying distinct adopter segments.

    Demonstrates using the MixtureModel to identify multiple distinct
    adoption patterns in heterogeneous data.
    """
    from innovate.diffuse.bass import BassModel
    from innovate.diffuse.logistic import LogisticModel
    from innovate.models.mixture import MixtureModel

    # Generate synthetic mixture data
    np.random.seed(42)
    t = np.linspace(0, 15, 60)

    # Component 1: Early adopters (fast Bass)
    p1, q1, m1 = 0.05, 0.5, 50.0
    y1 = m1 * (1 - np.exp(-(p1 + q1) * t)) / (1 + (q1 / p1) * np.exp(-(p1 + q1) * t))

    # Component 2: Late adopters (slow Logistic)
    y2 = 50.0 / (1 + np.exp(-0.3 * (t - 10.0)))

    # Combine with weights
    y_true = 0.4 * y1 + 0.6 * y2
    y_noisy = y_true + np.random.normal(0, 1.5, len(t))
    y_noisy = np.maximum(y_noisy, 0)

    # Fit mixture model
    model = MixtureModel(
        models=[BassModel(), LogisticModel()],
        max_iter=50,
    )
    model.fit(t, y_noisy)

    print("Mixture model results:")
    print(f"\nComponent weights: {model.weights}")

    print("\nFitted parameters:")
    for name, value in model.params_.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nR² score: {model.score(t, y_noisy):.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Diffusion Curve Fitting")
    print("=" * 60)
    example_basic_fitting()

    print("\n" + "=" * 60)
    print("Example 2: Model Comparison and Selection")
    print("=" * 60)
    example_model_comparison()

    print("\n" + "=" * 60)
    print("Example 3: Optimization Methods Comparison")
    print("=" * 60)
    example_optimization_methods()

    print("\n" + "=" * 60)
    print("Example 4: Fit Diagnostics and Residual Analysis")
    print("=" * 60)
    example_fit_diagnostics()

    print("\n" + "=" * 60)
    print("Example 5: Bootstrap Confidence Intervals")
    print("=" * 60)
    example_bootstrap_confidence_intervals()

    print("\n" + "=" * 60)
    print("Example 6: Mixture Models for Segment Identification")
    print("=" * 60)
    example_mixture_model()
