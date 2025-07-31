"""
Analysis functions for identifying reducing time series trends.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import ruptures as rpt
import pymannkendall as mk


def smooth_series(series, fraction=0.1):
    """
    Smooths a time series using LOESS.

    Args:
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns:
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def find_changepoint(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """
    Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns:
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=model).fit(series)

    if search_method == rpt.Pelt:
        # Pelt uses a penalty value
        result = algo.predict(pen=penalty_value)
    else:
        # Other methods like Binseg use n_bkps
        result = algo.predict(n_bkps=1)

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def verify_trend_decline(series):
    """
    Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns:
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def identify_reducing_series(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """
    Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns:
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                }
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = "no trend"
        p_value = 1.0
        slope = 0.0

        if changepoint_idx != -1 and changepoint_idx < len(smoothed) - 1:
            post_changepoint_series = smoothed[changepoint_idx:]
            if len(post_changepoint_series) >= 4:  # Check for MK test
                trend, p_value = verify_trend_decline(post_changepoint_series)

                # Calculate linear trend on post-changepoint data
                x = np.arange(len(post_changepoint_series))
                # Using np.polyfit for a simple linear regression
                coeffs = np.polyfit(x, post_changepoint_series, 1)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            }
        )

    return pd.DataFrame(results)
