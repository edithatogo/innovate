# src/innovate/preprocess/decomposition.py

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


def stl_decomposition(series: pd.Series, period: int, **kwargs):
    args = [series, period]# type: ignore
    kwargs = {**kwargs}# type: ignore
    return _mutmut_trampoline(x_stl_decomposition__mutmut_orig, x_stl_decomposition__mutmut_mutants, args, kwargs, None)


def x_stl_decomposition__mutmut_orig(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_1(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
    if isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The input series must have a DatetimeIndex.")

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_2(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
        raise TypeError(None)

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_3(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
        raise TypeError("XXThe input series must have a DatetimeIndex.XX")

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_4(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
        raise TypeError("the input series must have a datetimeindex.")

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_5(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
        raise TypeError("THE INPUT SERIES MUST HAVE A DATETIMEINDEX.")

    stl = STL(series, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_6(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = None
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_7(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = STL(None, period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_8(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = STL(series, period=None, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_9(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = STL(period=period, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_10(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = STL(series, **kwargs)
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_11(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    stl = STL(series, period=period, )
    result = stl.fit()

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_12(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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
    result = None

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_13(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        None,
    )


def x_stl_decomposition__mutmut_14(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "XXtrendXX": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_15(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "TREND": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_16(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "trend": result.trend,
            "XXseasonalXX": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_17(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "trend": result.trend,
            "SEASONAL": result.seasonal,
            "residual": result.resid,
        },
    )


def x_stl_decomposition__mutmut_18(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "XXresidualXX": result.resid,
        },
    )


def x_stl_decomposition__mutmut_19(series: pd.Series, period: int, **kwargs):
    """Decomposes a time series into trend, seasonal, and residual components
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

    return pd.DataFrame(
        {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "RESIDUAL": result.resid,
        },
    )

x_stl_decomposition__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_stl_decomposition__mutmut_1': x_stl_decomposition__mutmut_1, 
    'x_stl_decomposition__mutmut_2': x_stl_decomposition__mutmut_2, 
    'x_stl_decomposition__mutmut_3': x_stl_decomposition__mutmut_3, 
    'x_stl_decomposition__mutmut_4': x_stl_decomposition__mutmut_4, 
    'x_stl_decomposition__mutmut_5': x_stl_decomposition__mutmut_5, 
    'x_stl_decomposition__mutmut_6': x_stl_decomposition__mutmut_6, 
    'x_stl_decomposition__mutmut_7': x_stl_decomposition__mutmut_7, 
    'x_stl_decomposition__mutmut_8': x_stl_decomposition__mutmut_8, 
    'x_stl_decomposition__mutmut_9': x_stl_decomposition__mutmut_9, 
    'x_stl_decomposition__mutmut_10': x_stl_decomposition__mutmut_10, 
    'x_stl_decomposition__mutmut_11': x_stl_decomposition__mutmut_11, 
    'x_stl_decomposition__mutmut_12': x_stl_decomposition__mutmut_12, 
    'x_stl_decomposition__mutmut_13': x_stl_decomposition__mutmut_13, 
    'x_stl_decomposition__mutmut_14': x_stl_decomposition__mutmut_14, 
    'x_stl_decomposition__mutmut_15': x_stl_decomposition__mutmut_15, 
    'x_stl_decomposition__mutmut_16': x_stl_decomposition__mutmut_16, 
    'x_stl_decomposition__mutmut_17': x_stl_decomposition__mutmut_17, 
    'x_stl_decomposition__mutmut_18': x_stl_decomposition__mutmut_18, 
    'x_stl_decomposition__mutmut_19': x_stl_decomposition__mutmut_19
}
x_stl_decomposition__mutmut_orig.__name__ = 'x_stl_decomposition'
