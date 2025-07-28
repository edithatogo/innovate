import pandas as pd
import numpy as np
from typing import Dict, Any, List, Sequence, Tuple
from innovate.base.base import DiffusionModel
from .metrics import (
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_r_squared,
    calculate_mape,
    calculate_smape,
    calculate_rss,
    calculate_aic,
    calculate_bic,
)
from statsmodels.tsa.stattools import acf, pacf


def model_aic(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Akaike Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_aic(n_params, n_samples, rss)


def model_bic(model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> float:
    """Return the Bayesian Information Criterion for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
    y_pred = model.predict(t)
    rss = calculate_rss(y, y_pred)
    n_samples = len(y)
    n_params = len(model.param_names) + 1
    return calculate_bic(n_params, n_samples, rss)


def get_fit_metrics(
    model: DiffusionModel, t: Sequence[float], y: Sequence[float]
) -> Dict[str, float]:
    """
    Calculates various goodness-of-fit metrics for a model.

    Args:
        model: The fitted diffusion model.
        t: The time points.
        y: The true cumulative adoption values.

    Returns:
        A dictionary containing the calculated metrics.
    """
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)

    n_samples = len(y)
    # Add 1 to n_params for the variance of the residuals
    n_params = len(model.param_names) + 1

    rss = calculate_rss(y, y_pred)

    metrics = {
        "MSE": calculate_mse(y, y_pred),
        "RMSE": calculate_rmse(y, y_pred),
        "MAE": calculate_mae(y, y_pred),
        "R-squared": calculate_r_squared(y, y_pred),
        "MAPE": calculate_mape(y, y_pred),
        "SMAPE": calculate_smape(y, y_pred),
        "RSS": rss,
        "AIC": calculate_aic(n_params, n_samples, rss),
        "BIC": calculate_bic(n_params, n_samples, rss),
    }
    return metrics


def compare_models(
    models: Dict[str, DiffusionModel], t_true: Sequence[float], y_true: Sequence[float]
) -> pd.DataFrame:
    """
    Compares multiple diffusion models based on various goodness-of-fit metrics.

    Args:
        models: A dictionary where keys are model names (str) and values are
                fitted DiffusionModel instances.
        t_true: The true time points.
        y_true: The true cumulative adoption values.

    Returns:
        A pandas DataFrame containing the comparison metrics for each model.
    """
    results = []
    for name, model in models.items():
        if not hasattr(model, "predict") or not callable(model.predict):
            print(
                f"Warning: Model '{name}' does not have a 'predict' method. Skipping."
            )
            continue

        try:
            metrics = get_fit_metrics(model, t_true, y_true)
            metrics["Parameters"] = model.params_
            metrics["Model"] = name
            results.append(metrics)

        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index("Model")


def find_best_model(
    comparison_df: pd.DataFrame, metric: str = "RMSE", minimize: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Identifies the best performing model from a comparison DataFrame.

    Args:
        comparison_df: The DataFrame returned by compare_models.
        metric: The metric to use for comparison (e.g., 'RMSE', 'R-squared').
        minimize: If True, the best model has the minimum value for the metric.
                  If False, the best model has the maximum value.

    Returns:
        A tuple containing the name of the best model and its full results row.
    """
    if metric not in comparison_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in comparison DataFrame columns."
        )

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]

    return best_model_row.name, best_model_row.to_dict()


def compute_residuals(
    model: DiffusionModel, t: Sequence[float], y: Sequence[float]
) -> np.ndarray:
    """Return the residuals for a fitted model."""
    if not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    y_pred = model.predict(t)
    return np.asarray(y) - np.asarray(y_pred)


def residual_acf(
    model: DiffusionModel, t: Sequence[float], y: Sequence[float], nlags: int = 40
) -> np.ndarray:
    """Return the autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return acf(residuals, nlags=nlags)


def residual_pacf(
    model: DiffusionModel, t: Sequence[float], y: Sequence[float], nlags: int = 40
) -> np.ndarray:
    """Return the partial autocorrelation function of model residuals."""
    residuals = compute_residuals(model, t, y)
    return pacf(residuals, nlags=nlags)
