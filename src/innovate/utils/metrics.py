from collections.abc import Sequence

import numpy as np


def calculate_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def calculate_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def calculate_r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def calculate_smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def calculate_rss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def calculate_aic(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def calculate_bic(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)
