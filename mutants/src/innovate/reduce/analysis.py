"""Analysis functions for identifying reducing time series trends."""

import numpy as np
import pandas as pd
import pymannkendall as mk
import ruptures as rpt
import statsmodels.api as sm
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


def smooth_series(series, fraction=0.1):
    args = [series, fraction]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_smooth_series__mutmut_orig, x_smooth_series__mutmut_mutants, args, kwargs, None)


def x_smooth_series__mutmut_orig(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_1(series, fraction=1.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_2(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None and len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_3(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is not None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_4(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) != 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_5(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 1:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_6(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array(None)
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_7(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = None
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_8(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(None)
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_9(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = None
    return lowess[:, 1]


def x_smooth_series__mutmut_10(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(None, x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_11(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, None, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_12(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=None)
    return lowess[:, 1]


def x_smooth_series__mutmut_13(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(x, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_14(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, frac=fraction)
    return lowess[:, 1]


def x_smooth_series__mutmut_15(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, )
    return lowess[:, 1]


def x_smooth_series__mutmut_16(series, fraction=0.1):
    """Smooths a time series using LOESS.

    Args:
    ----
        series (np.array): The time series data.
        fraction (float): The fraction of data used when estimating each y-value.

    Returns
    -------
        np.array: The smoothed time series.
    """
    if series is None or len(series) == 0:
        return np.array([])
    x = np.arange(len(series))
    lowess = sm.nonparametric.lowess(series, x, frac=fraction)
    return lowess[:, 2]

x_smooth_series__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_smooth_series__mutmut_1': x_smooth_series__mutmut_1, 
    'x_smooth_series__mutmut_2': x_smooth_series__mutmut_2, 
    'x_smooth_series__mutmut_3': x_smooth_series__mutmut_3, 
    'x_smooth_series__mutmut_4': x_smooth_series__mutmut_4, 
    'x_smooth_series__mutmut_5': x_smooth_series__mutmut_5, 
    'x_smooth_series__mutmut_6': x_smooth_series__mutmut_6, 
    'x_smooth_series__mutmut_7': x_smooth_series__mutmut_7, 
    'x_smooth_series__mutmut_8': x_smooth_series__mutmut_8, 
    'x_smooth_series__mutmut_9': x_smooth_series__mutmut_9, 
    'x_smooth_series__mutmut_10': x_smooth_series__mutmut_10, 
    'x_smooth_series__mutmut_11': x_smooth_series__mutmut_11, 
    'x_smooth_series__mutmut_12': x_smooth_series__mutmut_12, 
    'x_smooth_series__mutmut_13': x_smooth_series__mutmut_13, 
    'x_smooth_series__mutmut_14': x_smooth_series__mutmut_14, 
    'x_smooth_series__mutmut_15': x_smooth_series__mutmut_15, 
    'x_smooth_series__mutmut_16': x_smooth_series__mutmut_16
}
x_smooth_series__mutmut_orig.__name__ = 'x_smooth_series'


def find_changepoint(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    args = [series, model, search_method, penalty_value]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_find_changepoint__mutmut_orig, x_find_changepoint__mutmut_mutants, args, kwargs, None)


def x_find_changepoint__mutmut_orig(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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


def x_find_changepoint__mutmut_1(series, model="XXl2XX", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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


def x_find_changepoint__mutmut_2(series, model="L2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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


def x_find_changepoint__mutmut_3(series, model="l2", search_method=rpt.Pelt, penalty_value=4):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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


def x_find_changepoint__mutmut_4(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None and len(series) < 2:
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


def x_find_changepoint__mutmut_5(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is not None or len(series) < 2:
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


def x_find_changepoint__mutmut_6(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) <= 2:
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


def x_find_changepoint__mutmut_7(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 3:
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


def x_find_changepoint__mutmut_8(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return +1

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


def x_find_changepoint__mutmut_9(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -2

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


def x_find_changepoint__mutmut_10(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = None

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


def x_find_changepoint__mutmut_11(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=model).fit(None)

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


def x_find_changepoint__mutmut_12(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=None).fit(series)

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


def x_find_changepoint__mutmut_13(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=model).fit(series)

    if search_method != rpt.Pelt:
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


def x_find_changepoint__mutmut_14(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=model).fit(series)

    if search_method == rpt.Pelt:
        # Pelt uses a penalty value
        result = None
    else:
        # Other methods like Binseg use n_bkps
        result = algo.predict(n_bkps=1)

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_15(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
        int: The index of the most likely changepoint. Returns -1 if no changepoint is found.
    """
    if series is None or len(series) < 2:
        return -1

    algo = search_method(model=model).fit(series)

    if search_method == rpt.Pelt:
        # Pelt uses a penalty value
        result = algo.predict(pen=None)
    else:
        # Other methods like Binseg use n_bkps
        result = algo.predict(n_bkps=1)

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_16(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
        result = None

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_17(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
        result = algo.predict(n_bkps=None)

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_18(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
        result = algo.predict(n_bkps=2)

    # result for 1 breakpoint is a list like [changepoint_index, end_of_series_index]
    if result and len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_19(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
    if result or len(result) > 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_20(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
    if result and len(result) >= 1:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_21(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
    if result and len(result) > 2:
        # For Pelt, result can have more than one breakpoint, we take the first one
        return result[0]
    return -1


def x_find_changepoint__mutmut_22(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
        return result[1]
    return -1


def x_find_changepoint__mutmut_23(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
    return +1


def x_find_changepoint__mutmut_24(series, model="l2", search_method=rpt.Pelt, penalty_value=3):
    """Finds the most likely single changepoint in a time series.

    This is useful for identifying the "peak" or the point where the
    trend begins to change.

    Args:
    ----
        series (np.array): The time series data.
        model (str): The model to use for changepoint detection (e.g., "l1", "l2").
        search_method (class): The ruptures search method to use (e.g., Pelt, Binseg).
        penalty_value (int): The penalty value for the Pelt search method.

    Returns
    -------
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
    return -2

x_find_changepoint__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_find_changepoint__mutmut_1': x_find_changepoint__mutmut_1, 
    'x_find_changepoint__mutmut_2': x_find_changepoint__mutmut_2, 
    'x_find_changepoint__mutmut_3': x_find_changepoint__mutmut_3, 
    'x_find_changepoint__mutmut_4': x_find_changepoint__mutmut_4, 
    'x_find_changepoint__mutmut_5': x_find_changepoint__mutmut_5, 
    'x_find_changepoint__mutmut_6': x_find_changepoint__mutmut_6, 
    'x_find_changepoint__mutmut_7': x_find_changepoint__mutmut_7, 
    'x_find_changepoint__mutmut_8': x_find_changepoint__mutmut_8, 
    'x_find_changepoint__mutmut_9': x_find_changepoint__mutmut_9, 
    'x_find_changepoint__mutmut_10': x_find_changepoint__mutmut_10, 
    'x_find_changepoint__mutmut_11': x_find_changepoint__mutmut_11, 
    'x_find_changepoint__mutmut_12': x_find_changepoint__mutmut_12, 
    'x_find_changepoint__mutmut_13': x_find_changepoint__mutmut_13, 
    'x_find_changepoint__mutmut_14': x_find_changepoint__mutmut_14, 
    'x_find_changepoint__mutmut_15': x_find_changepoint__mutmut_15, 
    'x_find_changepoint__mutmut_16': x_find_changepoint__mutmut_16, 
    'x_find_changepoint__mutmut_17': x_find_changepoint__mutmut_17, 
    'x_find_changepoint__mutmut_18': x_find_changepoint__mutmut_18, 
    'x_find_changepoint__mutmut_19': x_find_changepoint__mutmut_19, 
    'x_find_changepoint__mutmut_20': x_find_changepoint__mutmut_20, 
    'x_find_changepoint__mutmut_21': x_find_changepoint__mutmut_21, 
    'x_find_changepoint__mutmut_22': x_find_changepoint__mutmut_22, 
    'x_find_changepoint__mutmut_23': x_find_changepoint__mutmut_23, 
    'x_find_changepoint__mutmut_24': x_find_changepoint__mutmut_24
}
x_find_changepoint__mutmut_orig.__name__ = 'x_find_changepoint'


def verify_trend_decline(series):
    args = [series]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_verify_trend_decline__mutmut_orig, x_verify_trend_decline__mutmut_mutants, args, kwargs, None)


def x_verify_trend_decline__mutmut_orig(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_1(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None and len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_2(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is not None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_3(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) <= 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_4(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 5:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_5(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "XXno trendXX", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_6(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "NO TREND", 1.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_7(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 2.0
    test_result = mk.original_test(series)
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_8(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = None
    return test_result.trend, test_result.p


def x_verify_trend_decline__mutmut_9(series):
    """Verifies if a time series has a statistically significant decreasing trend
    using the Mann-Kendall test.

    Args:
    ----
        series (np.array): The time series data, typically the post-changepoint segment.

    Returns
    -------
        tuple: A tuple containing the trend result ('decreasing', 'increasing', 'no trend')
               and the p-value.
    """
    if series is None or len(series) < 4:  # Mann-Kendall needs at least 4 points
        return "no trend", 1.0
    test_result = mk.original_test(None)
    return test_result.trend, test_result.p

x_verify_trend_decline__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_verify_trend_decline__mutmut_1': x_verify_trend_decline__mutmut_1, 
    'x_verify_trend_decline__mutmut_2': x_verify_trend_decline__mutmut_2, 
    'x_verify_trend_decline__mutmut_3': x_verify_trend_decline__mutmut_3, 
    'x_verify_trend_decline__mutmut_4': x_verify_trend_decline__mutmut_4, 
    'x_verify_trend_decline__mutmut_5': x_verify_trend_decline__mutmut_5, 
    'x_verify_trend_decline__mutmut_6': x_verify_trend_decline__mutmut_6, 
    'x_verify_trend_decline__mutmut_7': x_verify_trend_decline__mutmut_7, 
    'x_verify_trend_decline__mutmut_8': x_verify_trend_decline__mutmut_8, 
    'x_verify_trend_decline__mutmut_9': x_verify_trend_decline__mutmut_9
}
x_verify_trend_decline__mutmut_orig.__name__ = 'x_verify_trend_decline'


def identify_reducing_series(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    args = [time_series_list, smooth_frac, changepoint_model, search_method, penalty_value]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_identify_reducing_series__mutmut_orig, x_identify_reducing_series__mutmut_mutants, args, kwargs, None)


def x_identify_reducing_series__mutmut_orig(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_1(
    time_series_list,
    smooth_frac=1.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_2(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="XXl2XX",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_3(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="L2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_4(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=4,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_5(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = None
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_6(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(None):
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
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_7(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = None

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_8(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(None, fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_9(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, fraction=None)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_10(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_11(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, )

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_12(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) <= 2:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_13(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 3:
            results.append(
                {
                    "series_index": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_14(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
        pd.DataFrame: A DataFrame summarizing the analysis for each time series,
                      with columns for changepoint index, trend result, and p-value.
    """
    results = []
    for i, series in enumerate(time_series_list):
        smoothed = smooth_series(series, fraction=smooth_frac)

        # Ensure we have enough data to find a changepoint
        if len(smoothed) < 2:
            results.append(
                None,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_15(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "XXseries_indexXX": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_16(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "SERIES_INDEX": i,
                    "changepoint_index": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_17(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "XXchangepoint_indexXX": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_18(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "CHANGEPOINT_INDEX": -1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_19(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "changepoint_index": +1,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_20(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "changepoint_index": -2,
                    "trend": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_21(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "XXtrendXX": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_22(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "TREND": "no trend",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_23(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "trend": "XXno trendXX",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_24(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "trend": "NO TREND",
                    "p_value": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_25(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "XXp_valueXX": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_26(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "P_VALUE": 1.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_27(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "p_value": 2.0,
                    "post_peak_slope": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_28(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "XXpost_peak_slopeXX": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_29(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "POST_PEAK_SLOPE": 0.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_30(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                    "post_peak_slope": 1.0,
                },
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_31(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            break

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_32(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = None

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_33(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            None,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_34(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=None,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_35(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=None,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_36(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=None,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_37(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_38(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_39(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_40(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_41(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = None
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_42(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = "XXno trendXX"
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_43(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = "NO TREND"
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_44(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = "no trend"
        p_value = None
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_45(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
            )
            continue

        changepoint_idx = find_changepoint(
            smoothed,
            model=changepoint_model,
            search_method=search_method,
            penalty_value=penalty_value,
        )

        trend = "no trend"
        p_value = 2.0
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_46(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
        slope = None

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_47(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
        slope = 1.0

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_48(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != -1 or changepoint_idx < len(smoothed) - 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_49(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx == -1 and changepoint_idx < len(smoothed) - 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_50(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != +1 and changepoint_idx < len(smoothed) - 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_51(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != -2 and changepoint_idx < len(smoothed) - 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_52(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != -1 and changepoint_idx <= len(smoothed) - 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_53(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != -1 and changepoint_idx < len(smoothed) + 1:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_54(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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

        if changepoint_idx != -1 and changepoint_idx < len(smoothed) - 2:
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_55(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            post_changepoint_series = None
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_56(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            if len(post_changepoint_series) > 4:  # Check for MK test
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_57(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            if len(post_changepoint_series) >= 5:  # Check for MK test
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_58(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                trend, p_value = None

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_59(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                trend, p_value = verify_trend_decline(None)

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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_60(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                x = None
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_61(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                x = np.arange(None)
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
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_62(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = None
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_63(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(None, post_changepoint_series, 1)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_64(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(x, None, 1)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_65(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(x, post_changepoint_series, None)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_66(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(post_changepoint_series, 1)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_67(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(x, 1)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_68(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(x, post_changepoint_series, )
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_69(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                coeffs = np.polyfit(x, post_changepoint_series, 2)
                slope = coeffs[0]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_70(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                slope = None

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_71(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                slope = coeffs[1]

        results.append(
            {
                "series_index": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_72(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            None,
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_73(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "XXseries_indexXX": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_74(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "SERIES_INDEX": i,
                "changepoint_index": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_75(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "XXchangepoint_indexXX": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_76(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "CHANGEPOINT_INDEX": changepoint_idx,
                "trend": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_77(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "XXtrendXX": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_78(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "TREND": trend,
                "p_value": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_79(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "XXp_valueXX": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_80(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "P_VALUE": p_value,
                "post_peak_slope": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_81(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "XXpost_peak_slopeXX": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_82(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
                "POST_PEAK_SLOPE": slope,
            },
        )

    return pd.DataFrame(results)


def x_identify_reducing_series__mutmut_83(
    time_series_list,
    smooth_frac=0.1,
    changepoint_model="l2",
    search_method=rpt.Binseg,
    penalty_value=3,
):
    """Analyzes a list of time series to identify those with a reducing trend.

    This function acts as a pipeline:
    1. Smooths each series.
    2. Finds the most likely changepoint (peak).
    3. Performs a Mann-Kendall test on the post-changepoint data.

    Args:
    ----
        time_series_list (list of np.array): A list of time series to analyze.
        smooth_frac (float): The fraction for the LOESS smoother.
        changepoint_model (str): The model for changepoint detection.
        search_method (class): The ruptures search method to use.
        penalty_value (int): The penalty value for the Pelt search method (if used).

    Returns
    -------
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
                },
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
            },
        )

    return pd.DataFrame(None)

x_identify_reducing_series__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_identify_reducing_series__mutmut_1': x_identify_reducing_series__mutmut_1, 
    'x_identify_reducing_series__mutmut_2': x_identify_reducing_series__mutmut_2, 
    'x_identify_reducing_series__mutmut_3': x_identify_reducing_series__mutmut_3, 
    'x_identify_reducing_series__mutmut_4': x_identify_reducing_series__mutmut_4, 
    'x_identify_reducing_series__mutmut_5': x_identify_reducing_series__mutmut_5, 
    'x_identify_reducing_series__mutmut_6': x_identify_reducing_series__mutmut_6, 
    'x_identify_reducing_series__mutmut_7': x_identify_reducing_series__mutmut_7, 
    'x_identify_reducing_series__mutmut_8': x_identify_reducing_series__mutmut_8, 
    'x_identify_reducing_series__mutmut_9': x_identify_reducing_series__mutmut_9, 
    'x_identify_reducing_series__mutmut_10': x_identify_reducing_series__mutmut_10, 
    'x_identify_reducing_series__mutmut_11': x_identify_reducing_series__mutmut_11, 
    'x_identify_reducing_series__mutmut_12': x_identify_reducing_series__mutmut_12, 
    'x_identify_reducing_series__mutmut_13': x_identify_reducing_series__mutmut_13, 
    'x_identify_reducing_series__mutmut_14': x_identify_reducing_series__mutmut_14, 
    'x_identify_reducing_series__mutmut_15': x_identify_reducing_series__mutmut_15, 
    'x_identify_reducing_series__mutmut_16': x_identify_reducing_series__mutmut_16, 
    'x_identify_reducing_series__mutmut_17': x_identify_reducing_series__mutmut_17, 
    'x_identify_reducing_series__mutmut_18': x_identify_reducing_series__mutmut_18, 
    'x_identify_reducing_series__mutmut_19': x_identify_reducing_series__mutmut_19, 
    'x_identify_reducing_series__mutmut_20': x_identify_reducing_series__mutmut_20, 
    'x_identify_reducing_series__mutmut_21': x_identify_reducing_series__mutmut_21, 
    'x_identify_reducing_series__mutmut_22': x_identify_reducing_series__mutmut_22, 
    'x_identify_reducing_series__mutmut_23': x_identify_reducing_series__mutmut_23, 
    'x_identify_reducing_series__mutmut_24': x_identify_reducing_series__mutmut_24, 
    'x_identify_reducing_series__mutmut_25': x_identify_reducing_series__mutmut_25, 
    'x_identify_reducing_series__mutmut_26': x_identify_reducing_series__mutmut_26, 
    'x_identify_reducing_series__mutmut_27': x_identify_reducing_series__mutmut_27, 
    'x_identify_reducing_series__mutmut_28': x_identify_reducing_series__mutmut_28, 
    'x_identify_reducing_series__mutmut_29': x_identify_reducing_series__mutmut_29, 
    'x_identify_reducing_series__mutmut_30': x_identify_reducing_series__mutmut_30, 
    'x_identify_reducing_series__mutmut_31': x_identify_reducing_series__mutmut_31, 
    'x_identify_reducing_series__mutmut_32': x_identify_reducing_series__mutmut_32, 
    'x_identify_reducing_series__mutmut_33': x_identify_reducing_series__mutmut_33, 
    'x_identify_reducing_series__mutmut_34': x_identify_reducing_series__mutmut_34, 
    'x_identify_reducing_series__mutmut_35': x_identify_reducing_series__mutmut_35, 
    'x_identify_reducing_series__mutmut_36': x_identify_reducing_series__mutmut_36, 
    'x_identify_reducing_series__mutmut_37': x_identify_reducing_series__mutmut_37, 
    'x_identify_reducing_series__mutmut_38': x_identify_reducing_series__mutmut_38, 
    'x_identify_reducing_series__mutmut_39': x_identify_reducing_series__mutmut_39, 
    'x_identify_reducing_series__mutmut_40': x_identify_reducing_series__mutmut_40, 
    'x_identify_reducing_series__mutmut_41': x_identify_reducing_series__mutmut_41, 
    'x_identify_reducing_series__mutmut_42': x_identify_reducing_series__mutmut_42, 
    'x_identify_reducing_series__mutmut_43': x_identify_reducing_series__mutmut_43, 
    'x_identify_reducing_series__mutmut_44': x_identify_reducing_series__mutmut_44, 
    'x_identify_reducing_series__mutmut_45': x_identify_reducing_series__mutmut_45, 
    'x_identify_reducing_series__mutmut_46': x_identify_reducing_series__mutmut_46, 
    'x_identify_reducing_series__mutmut_47': x_identify_reducing_series__mutmut_47, 
    'x_identify_reducing_series__mutmut_48': x_identify_reducing_series__mutmut_48, 
    'x_identify_reducing_series__mutmut_49': x_identify_reducing_series__mutmut_49, 
    'x_identify_reducing_series__mutmut_50': x_identify_reducing_series__mutmut_50, 
    'x_identify_reducing_series__mutmut_51': x_identify_reducing_series__mutmut_51, 
    'x_identify_reducing_series__mutmut_52': x_identify_reducing_series__mutmut_52, 
    'x_identify_reducing_series__mutmut_53': x_identify_reducing_series__mutmut_53, 
    'x_identify_reducing_series__mutmut_54': x_identify_reducing_series__mutmut_54, 
    'x_identify_reducing_series__mutmut_55': x_identify_reducing_series__mutmut_55, 
    'x_identify_reducing_series__mutmut_56': x_identify_reducing_series__mutmut_56, 
    'x_identify_reducing_series__mutmut_57': x_identify_reducing_series__mutmut_57, 
    'x_identify_reducing_series__mutmut_58': x_identify_reducing_series__mutmut_58, 
    'x_identify_reducing_series__mutmut_59': x_identify_reducing_series__mutmut_59, 
    'x_identify_reducing_series__mutmut_60': x_identify_reducing_series__mutmut_60, 
    'x_identify_reducing_series__mutmut_61': x_identify_reducing_series__mutmut_61, 
    'x_identify_reducing_series__mutmut_62': x_identify_reducing_series__mutmut_62, 
    'x_identify_reducing_series__mutmut_63': x_identify_reducing_series__mutmut_63, 
    'x_identify_reducing_series__mutmut_64': x_identify_reducing_series__mutmut_64, 
    'x_identify_reducing_series__mutmut_65': x_identify_reducing_series__mutmut_65, 
    'x_identify_reducing_series__mutmut_66': x_identify_reducing_series__mutmut_66, 
    'x_identify_reducing_series__mutmut_67': x_identify_reducing_series__mutmut_67, 
    'x_identify_reducing_series__mutmut_68': x_identify_reducing_series__mutmut_68, 
    'x_identify_reducing_series__mutmut_69': x_identify_reducing_series__mutmut_69, 
    'x_identify_reducing_series__mutmut_70': x_identify_reducing_series__mutmut_70, 
    'x_identify_reducing_series__mutmut_71': x_identify_reducing_series__mutmut_71, 
    'x_identify_reducing_series__mutmut_72': x_identify_reducing_series__mutmut_72, 
    'x_identify_reducing_series__mutmut_73': x_identify_reducing_series__mutmut_73, 
    'x_identify_reducing_series__mutmut_74': x_identify_reducing_series__mutmut_74, 
    'x_identify_reducing_series__mutmut_75': x_identify_reducing_series__mutmut_75, 
    'x_identify_reducing_series__mutmut_76': x_identify_reducing_series__mutmut_76, 
    'x_identify_reducing_series__mutmut_77': x_identify_reducing_series__mutmut_77, 
    'x_identify_reducing_series__mutmut_78': x_identify_reducing_series__mutmut_78, 
    'x_identify_reducing_series__mutmut_79': x_identify_reducing_series__mutmut_79, 
    'x_identify_reducing_series__mutmut_80': x_identify_reducing_series__mutmut_80, 
    'x_identify_reducing_series__mutmut_81': x_identify_reducing_series__mutmut_81, 
    'x_identify_reducing_series__mutmut_82': x_identify_reducing_series__mutmut_82, 
    'x_identify_reducing_series__mutmut_83': x_identify_reducing_series__mutmut_83
}
x_identify_reducing_series__mutmut_orig.__name__ = 'x_identify_reducing_series'
