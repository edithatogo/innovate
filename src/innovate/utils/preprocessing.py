"""Preprocessing helpers for diffusion time series."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def ensure_datetime_index(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def aggregate_time_series(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = ensure_datetime_index(data)
    if freq == "W":
        # Treat weekly aggregation as contiguous 7-day windows from the first
        # observation rather than pandas' calendar week buckets.
        start = data.index.min()
        week_buckets = ((data.index - start) // pd.Timedelta(days=7)).astype(int)
        aggregated = data.groupby(week_buckets).sum()
        aggregated.index = start + pd.to_timedelta(aggregated.index * 7, unit="D")
        return aggregated
    return data.resample(freq).sum()


def apply_stl_decomposition(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply STL decomposition to a time series."""
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) > 12:
            period = 12  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid  # noqa: TRY300
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def cumulative_sum(data: Sequence[float]) -> np.ndarray:
    """Calculates the cumulative sum of a sequence."""
    return np.cumsum(data)


def apply_rolling_average(data: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to a time series."""
    return data.rolling(window=window).mean()


def apply_sarima(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues
