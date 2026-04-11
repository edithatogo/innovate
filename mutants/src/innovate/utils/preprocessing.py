from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


def ensure_datetime_index(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    args = [data]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_ensure_datetime_index__mutmut_orig, x_ensure_datetime_index__mutmut_mutants, args, kwargs, None)


def x_ensure_datetime_index__mutmut_orig(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def x_ensure_datetime_index__mutmut_1(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def x_ensure_datetime_index__mutmut_2(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = None
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def x_ensure_datetime_index__mutmut_3(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(None)
        except Exception as e:
            raise ValueError(f"Could not convert index to DatetimeIndex: {e}")
    return data


def x_ensure_datetime_index__mutmut_4(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Ensures a pandas Series or DataFrame has a datetime index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(None)
    return data

x_ensure_datetime_index__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_ensure_datetime_index__mutmut_1': x_ensure_datetime_index__mutmut_1, 
    'x_ensure_datetime_index__mutmut_2': x_ensure_datetime_index__mutmut_2, 
    'x_ensure_datetime_index__mutmut_3': x_ensure_datetime_index__mutmut_3, 
    'x_ensure_datetime_index__mutmut_4': x_ensure_datetime_index__mutmut_4
}
x_ensure_datetime_index__mutmut_orig.__name__ = 'x_ensure_datetime_index'


def aggregate_time_series(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    args = [data, freq]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_aggregate_time_series__mutmut_orig, x_aggregate_time_series__mutmut_mutants, args, kwargs, None)


def x_aggregate_time_series__mutmut_orig(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = ensure_datetime_index(data)
    return data.resample(freq).sum()


def x_aggregate_time_series__mutmut_1(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = None
    return data.resample(freq).sum()


def x_aggregate_time_series__mutmut_2(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = ensure_datetime_index(None)
    return data.resample(freq).sum()


def x_aggregate_time_series__mutmut_3(
    data: pd.Series | pd.DataFrame,
    freq: str,
) -> pd.Series | pd.DataFrame:
    """Aggregates time series data to a specified frequency (e.g., 'D', 'W', 'M')."""
    data = ensure_datetime_index(data)
    return data.resample(None).sum()

x_aggregate_time_series__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_aggregate_time_series__mutmut_1': x_aggregate_time_series__mutmut_1, 
    'x_aggregate_time_series__mutmut_2': x_aggregate_time_series__mutmut_2, 
    'x_aggregate_time_series__mutmut_3': x_aggregate_time_series__mutmut_3
}
x_aggregate_time_series__mutmut_orig.__name__ = 'x_aggregate_time_series'


def apply_stl_decomposition(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    args = [data, period, robust]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_apply_stl_decomposition__mutmut_orig, x_apply_stl_decomposition__mutmut_mutants, args, kwargs, None)


def x_apply_stl_decomposition__mutmut_orig(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_1(
    data: pd.Series,
    period: int | None = None,
    robust: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_2(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = None
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
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_3(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(None)
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
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_4(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is not None:
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
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_5(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) >= 12:
            period = 12  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_6(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) > 13:
            period = 12  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_7(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) > 12:
            period = None  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_8(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
        A tuple of (trend, seasonal, residuals) as pandas Series.
    """
    data = ensure_datetime_index(data)
    if period is None:
        # Attempt to infer period if not provided
        # This is a basic heuristic; more sophisticated methods might be needed
        if len(data) > 12:
            period = 13  # Assume monthly seasonality if data is long enough
        else:
            raise ValueError(
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_9(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                None,
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_10(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "XXPeriod must be specified for STL decomposition if data length is too short for inference.XX",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_11(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "period must be specified for stl decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_12(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "PERIOD MUST BE SPECIFIED FOR STL DECOMPOSITION IF DATA LENGTH IS TOO SHORT FOR INFERENCE.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_13(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = None
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_14(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(None, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_15(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=None, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_16(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=None)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_17(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_18(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_19(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, )
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_20(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = None
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(f"STL decomposition failed: {e}")


def x_apply_stl_decomposition__mutmut_21(
    data: pd.Series,
    period: int | None = None,
    robust: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Applies Seasonal-Trend decomposition using Loess (STL) to a time series.

    Args:
    ----
        data: A pandas Series with a DatetimeIndex.
        period: Period of the seasonality. If None, it will try to infer.
        robust: Whether to use robust fitting (less sensitive to outliers).

    Returns
    -------
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
                "Period must be specified for STL decomposition if data length is too short for inference.",
            )

    try:
        stl = STL(data, period=period, robust=robust)
        res = stl.fit()
        return res.trend, res.seasonal, res.resid
    except Exception as e:
        raise RuntimeError(None)

x_apply_stl_decomposition__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_apply_stl_decomposition__mutmut_1': x_apply_stl_decomposition__mutmut_1, 
    'x_apply_stl_decomposition__mutmut_2': x_apply_stl_decomposition__mutmut_2, 
    'x_apply_stl_decomposition__mutmut_3': x_apply_stl_decomposition__mutmut_3, 
    'x_apply_stl_decomposition__mutmut_4': x_apply_stl_decomposition__mutmut_4, 
    'x_apply_stl_decomposition__mutmut_5': x_apply_stl_decomposition__mutmut_5, 
    'x_apply_stl_decomposition__mutmut_6': x_apply_stl_decomposition__mutmut_6, 
    'x_apply_stl_decomposition__mutmut_7': x_apply_stl_decomposition__mutmut_7, 
    'x_apply_stl_decomposition__mutmut_8': x_apply_stl_decomposition__mutmut_8, 
    'x_apply_stl_decomposition__mutmut_9': x_apply_stl_decomposition__mutmut_9, 
    'x_apply_stl_decomposition__mutmut_10': x_apply_stl_decomposition__mutmut_10, 
    'x_apply_stl_decomposition__mutmut_11': x_apply_stl_decomposition__mutmut_11, 
    'x_apply_stl_decomposition__mutmut_12': x_apply_stl_decomposition__mutmut_12, 
    'x_apply_stl_decomposition__mutmut_13': x_apply_stl_decomposition__mutmut_13, 
    'x_apply_stl_decomposition__mutmut_14': x_apply_stl_decomposition__mutmut_14, 
    'x_apply_stl_decomposition__mutmut_15': x_apply_stl_decomposition__mutmut_15, 
    'x_apply_stl_decomposition__mutmut_16': x_apply_stl_decomposition__mutmut_16, 
    'x_apply_stl_decomposition__mutmut_17': x_apply_stl_decomposition__mutmut_17, 
    'x_apply_stl_decomposition__mutmut_18': x_apply_stl_decomposition__mutmut_18, 
    'x_apply_stl_decomposition__mutmut_19': x_apply_stl_decomposition__mutmut_19, 
    'x_apply_stl_decomposition__mutmut_20': x_apply_stl_decomposition__mutmut_20, 
    'x_apply_stl_decomposition__mutmut_21': x_apply_stl_decomposition__mutmut_21
}
x_apply_stl_decomposition__mutmut_orig.__name__ = 'x_apply_stl_decomposition'


def cumulative_sum(data: Sequence[float]) -> np.ndarray:
    args = [data]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_cumulative_sum__mutmut_orig, x_cumulative_sum__mutmut_mutants, args, kwargs, None)


def x_cumulative_sum__mutmut_orig(data: Sequence[float]) -> np.ndarray:
    """Calculates the cumulative sum of a sequence."""
    return np.cumsum(data)


def x_cumulative_sum__mutmut_1(data: Sequence[float]) -> np.ndarray:
    """Calculates the cumulative sum of a sequence."""
    return np.cumsum(None)

x_cumulative_sum__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_cumulative_sum__mutmut_1': x_cumulative_sum__mutmut_1
}
x_cumulative_sum__mutmut_orig.__name__ = 'x_cumulative_sum'


def apply_rolling_average(data: pd.Series, window: int) -> pd.Series:
    args = [data, window]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_apply_rolling_average__mutmut_orig, x_apply_rolling_average__mutmut_mutants, args, kwargs, None)


def x_apply_rolling_average__mutmut_orig(data: pd.Series, window: int) -> pd.Series:
    """Applies a rolling average to a time series.

    Args:
    ----
        data: A pandas Series.
        window: The size of the rolling window.

    Returns
    -------
        A pandas Series with the rolling average applied.
    """
    return data.rolling(window=window).mean()


def x_apply_rolling_average__mutmut_1(data: pd.Series, window: int) -> pd.Series:
    """Applies a rolling average to a time series.

    Args:
    ----
        data: A pandas Series.
        window: The size of the rolling window.

    Returns
    -------
        A pandas Series with the rolling average applied.
    """
    return data.rolling(window=None).mean()

x_apply_rolling_average__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_apply_rolling_average__mutmut_1': x_apply_rolling_average__mutmut_1
}
x_apply_rolling_average__mutmut_orig.__name__ = 'x_apply_rolling_average'


def apply_sarima(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    args = [data, order, seasonal_order]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_apply_sarima__mutmut_orig, x_apply_sarima__mutmut_mutants, args, kwargs, None)


def x_apply_sarima__mutmut_orig(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_1(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = None
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_2(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(None, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_3(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=None, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_4(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=None)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_5(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_6(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_7(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, )
    results = model.fit(disp=False)
    return results.fittedvalues


def x_apply_sarima__mutmut_8(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = None
    return results.fittedvalues


def x_apply_sarima__mutmut_9(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=None)
    return results.fittedvalues


def x_apply_sarima__mutmut_10(
    data: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fits a SARIMA model to a time series and returns the fitted values.

    Args:
    ----
        data: A pandas Series.
        order: The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        seasonal_order: The (P,D,Q,s) seasonal order of the model.

    Returns
    -------
        A pandas Series with the fitted values from the SARIMA model.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=True)
    return results.fittedvalues

x_apply_sarima__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_apply_sarima__mutmut_1': x_apply_sarima__mutmut_1, 
    'x_apply_sarima__mutmut_2': x_apply_sarima__mutmut_2, 
    'x_apply_sarima__mutmut_3': x_apply_sarima__mutmut_3, 
    'x_apply_sarima__mutmut_4': x_apply_sarima__mutmut_4, 
    'x_apply_sarima__mutmut_5': x_apply_sarima__mutmut_5, 
    'x_apply_sarima__mutmut_6': x_apply_sarima__mutmut_6, 
    'x_apply_sarima__mutmut_7': x_apply_sarima__mutmut_7, 
    'x_apply_sarima__mutmut_8': x_apply_sarima__mutmut_8, 
    'x_apply_sarima__mutmut_9': x_apply_sarima__mutmut_9, 
    'x_apply_sarima__mutmut_10': x_apply_sarima__mutmut_10
}
x_apply_sarima__mutmut_orig.__name__ = 'x_apply_sarima'
