import pandas as pd
import numpy as np
from typing import Sequence, Tuple, Union
from statsmodels.tsa.seasonal import STL


def ensure_datetime_index(
    data: Union[pd.Series, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def aggregate_time_series(
    data: Union[pd.Series, pd.DataFrame], freq: str
) -> Union[pd.Series, pd.DataFrame]:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = ensure_datetime_index(data)
    return data.resample(freq).sum()


def apply_stl_decomposition(
    data: pd.Series, period: int = None, robust: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns:
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) > 12:
            period = 12  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference."
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def cumulative_sum(data: Sequence[float]) -> np.ndarray:
    """Calculates the cumulative sum of a sequence."""
    return np.cumsum(data)


def apply_rolling_average(data: pd.Series, window: int) -> pd.Series:
    """
    Applies a rolling average to a time series.

    Args:
        data: A pandas Series.
        window: The size of the rolling window.

    Returns:
        A pandas Series with the rolling average applied.
    """
    return data.rolling(window=window).mean()


def apply_sarima(
    data: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> pd.Series:
    """
    Fits a SARIMA model to a time series and returns the fitted values.

    Args:
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns:
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues
