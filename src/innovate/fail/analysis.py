# src/innovate/fail/analysis.py

import numpy as np
from typing import List

def analyze_failure(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> List[int]:
    """
    Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns:
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]
    
    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)
            
    return failed_indices