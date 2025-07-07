import numpy as np
from typing import Sequence

def calculate_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return np.mean((y_true_arr - y_pred_arr) ** 2)

def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))

def calculate_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return np.nan # Or raise an error, depending on desired behavior
    return np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100

def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return np.mean(np.abs(y_true_arr - y_pred_arr))

def calculate_r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def calculate_smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else np.nan
    return np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
