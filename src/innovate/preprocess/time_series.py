# src/innovate/preprocess/time_series.py

"""Convenience wrappers around utilities for common time-series preprocessing."""

from __future__ import annotations

import pandas as pd
from typing import Tuple

from innovate.utils.preprocessing import apply_rolling_average, apply_sarima


def rolling_average(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(series, window)


def sarima_fit(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, order=order, seasonal_order=seasonal_order)
