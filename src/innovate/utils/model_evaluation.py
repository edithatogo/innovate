import pandas as pd
import numpy as np
from typing import Dict, Any, List, Sequence, Tuple
from ..models.base import DiffusionModel
from .metrics import calculate_mse, calculate_rmse, calculate_mae, calculate_r_squared, calculate_mape, calculate_smape

def compare_models(
    models: Dict[str, DiffusionModel],
    t_true: Sequence[float],
    y_true: Sequence[float]
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
        if not hasattr(model, 'predict') or not callable(model.predict):
            print(f"Warning: Model '{name}' does not have a 'predict' method. Skipping.")
            continue
        
        try:
            y_pred = model.predict(t_true)
            
            mse = calculate_mse(y_true, y_pred)
            rmse = calculate_rmse(y_true, y_pred)
            mae = calculate_mae(y_true, y_pred)
            r_squared = calculate_r_squared(y_true, y_pred)
            mape = calculate_mape(y_true, y_pred)
            smape = calculate_smape(y_true, y_pred)

            results.append({
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R-squared': r_squared,
                'MAPE': mape,
                'SMAPE': smape,
                'Parameters': model.params_
            })
        except Exception as e:
            print(f"Error evaluating model '{name}': {e}. Skipping.")
            continue

    return pd.DataFrame(results).set_index('Model')

def find_best_model(
    comparison_df: pd.DataFrame,
    metric: str = 'RMSE',
    minimize: bool = True
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
        raise ValueError(f"Metric '{metric}' not found in comparison DataFrame columns.")

    if minimize:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmin()]
    else:
        best_model_row = comparison_df.loc[comparison_df[metric].idxmax()]
    
    return best_model_row.name, best_model_row.to_dict()
