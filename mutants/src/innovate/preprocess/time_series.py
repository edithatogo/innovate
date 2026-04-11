# src/innovate/preprocess/time_series.py

"""Convenience wrappers around utilities for common time-series preprocessing."""

from __future__ import annotations

import pandas as pd

from innovate.utils.preprocessing import apply_rolling_average, apply_sarima
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


def rolling_average(series: pd.Series, window: int) -> pd.Series:
    args = [series, window]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_rolling_average__mutmut_orig, x_rolling_average__mutmut_mutants, args, kwargs, None)


def x_rolling_average__mutmut_orig(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(series, window)


def x_rolling_average__mutmut_1(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(None, window)


def x_rolling_average__mutmut_2(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(series, None)


def x_rolling_average__mutmut_3(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(window)


def x_rolling_average__mutmut_4(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling average to ``series`` using ``window`` size."""
    return apply_rolling_average(series, )

x_rolling_average__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_rolling_average__mutmut_1': x_rolling_average__mutmut_1, 
    'x_rolling_average__mutmut_2': x_rolling_average__mutmut_2, 
    'x_rolling_average__mutmut_3': x_rolling_average__mutmut_3, 
    'x_rolling_average__mutmut_4': x_rolling_average__mutmut_4
}
x_rolling_average__mutmut_orig.__name__ = 'x_rolling_average'


def sarima_fit(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    args = [series, order, seasonal_order]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_sarima_fit__mutmut_orig, x_sarima_fit__mutmut_mutants, args, kwargs, None)


def x_sarima_fit__mutmut_orig(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, order=order, seasonal_order=seasonal_order)


def x_sarima_fit__mutmut_1(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(None, order=order, seasonal_order=seasonal_order)


def x_sarima_fit__mutmut_2(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, order=None, seasonal_order=seasonal_order)


def x_sarima_fit__mutmut_3(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, order=order, seasonal_order=None)


def x_sarima_fit__mutmut_4(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(order=order, seasonal_order=seasonal_order)


def x_sarima_fit__mutmut_5(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, seasonal_order=seasonal_order)


def x_sarima_fit__mutmut_6(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.Series:
    """Fit a SARIMA model and return the fitted values."""
    return apply_sarima(series, order=order, )

x_sarima_fit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_sarima_fit__mutmut_1': x_sarima_fit__mutmut_1, 
    'x_sarima_fit__mutmut_2': x_sarima_fit__mutmut_2, 
    'x_sarima_fit__mutmut_3': x_sarima_fit__mutmut_3, 
    'x_sarima_fit__mutmut_4': x_sarima_fit__mutmut_4, 
    'x_sarima_fit__mutmut_5': x_sarima_fit__mutmut_5, 
    'x_sarima_fit__mutmut_6': x_sarima_fit__mutmut_6
}
x_sarima_fit__mutmut_orig.__name__ = 'x_sarima_fit'
