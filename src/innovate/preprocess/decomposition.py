# src/innovate/preprocess/decomposition.py

import pandas as pd
from statsmodels.tsa.seasonal import STL

def stl_decomposition(series: pd.Series, period: int, **kwargs):
    """
    Decomposes a time series into trend, seasonal, and residual components
    using STL (Seasonal and Trend decomposition using Loess).

    Parameters
    ----------
    series : pd.Series
        The time series to decompose. Must have a DatetimeIndex.
    period : int
        The seasonal period of the time series.
    kwargs : dict
        Additional keyword arguments to pass to the STL function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the trend, seasonal, and residual components.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The input series must have a DatetimeIndex.")

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()
    
    return pd.DataFrame({
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
    })
