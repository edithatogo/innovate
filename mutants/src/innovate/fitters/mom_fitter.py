from collections.abc import Sequence

import numpy as np
import pandas as pd

from innovate.base.base import DiffusionModel
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


def estimate_bass_mom(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    args = [t, y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_estimate_bass_mom__mutmut_orig, x_estimate_bass_mom__mutmut_mutants, args, kwargs, None)


def x_estimate_bass_mom__mutmut_orig(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_1(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) and len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_2(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) == len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_3(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) <= 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_4(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 4:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_5(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            None,
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_6(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "XXInput sequences t and y must have the same length and at least 3 data points.XX",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_7(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_8(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "INPUT SEQUENCES T AND Y MUST HAVE THE SAME LENGTH AND AT LEAST 3 DATA POINTS.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_9(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = None

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_10(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(None, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_11(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=None)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_12(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_13(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, )

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_14(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = None

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_15(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(None)

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_16(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[1])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_17(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = None

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_18(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(None)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_19(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(None).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_20(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(2).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_21(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(1)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_22(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = None

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_23(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        None,
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_24(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"XXx_tXX": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_25(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"X_T": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_26(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "XXy_t_minus_1XX": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_27(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "Y_T_MINUS_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_28(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = None

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_29(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[2:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_30(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) <= 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_31(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 4:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_32(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            None,
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_33(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "XXNot enough valid data points for Bass MoM estimation after preprocessing.XX",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_34(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "not enough valid data points for bass mom estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_35(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "NOT ENOUGH VALID DATA POINTS FOR BASS MOM ESTIMATION AFTER PREPROCESSING.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_36(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = None
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_37(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        None,
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_38(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "XXinterceptXX": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_39(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "INTERCEPT": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_40(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 2,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_41(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "XXy_t_minus_1XX": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_42(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "Y_T_MINUS_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_43(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["XXy_t_minus_1XX"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_44(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["Y_T_MINUS_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_45(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "XXy_t_minus_1_sqXX": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_46(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "Y_T_MINUS_1_SQ": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_47(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] * 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_48(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["XXy_t_minus_1XX"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_49(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["Y_T_MINUS_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_50(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 3,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_51(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = None

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_52(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["XXx_tXX"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_53(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["X_T"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_54(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = None
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_55(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(None, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_56(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, None, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_57(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_58(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_59(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, )[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_60(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[1]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_61(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            None,
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_62(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = None

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_63(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[1], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_64(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[2], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_65(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[3]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_66(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = None

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_67(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 + 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_68(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b * 2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_69(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**3 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_70(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a / c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_71(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 / a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_72(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 5 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_73(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant <= 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_74(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 1:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_75(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            None,
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_76(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "XXDiscriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.XX",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_77(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "discriminant is negative, no real solution for m. bass mom estimation failed. data might not fit bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_78(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "DISCRIMINANT IS NEGATIVE, NO REAL SOLUTION FOR M. BASS MOM ESTIMATION FAILED. DATA MIGHT NOT FIT BASS MODEL ASSUMPTIONS WELL.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_79(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = None
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_80(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) * (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_81(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b - np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_82(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (+b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_83(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(None)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_84(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 / c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_85(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (3 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_86(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = None

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_87(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) * (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_88(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b + np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_89(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (+b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_90(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(None)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_91(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 / c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_92(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (3 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_93(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = None

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_94(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 or val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_95(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val >= 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_96(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 1 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_97(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val > np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_98(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(None)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_99(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_100(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            None,
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_101(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "XXNo valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.XX",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_102(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "no valid positive market potential (m) found that is greater than or equal to max observed adoption. bass mom estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_103(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "NO VALID POSITIVE MARKET POTENTIAL (M) FOUND THAT IS GREATER THAN OR EQUAL TO MAX OBSERVED ADOPTION. BASS MOM ESTIMATION FAILED.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_104(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = None

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_105(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(None)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_106(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = None
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_107(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c / m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_108(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = +c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_109(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = None

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_110(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a * m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_111(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 and q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_112(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p < 0 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_113(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 1 or q <= 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_114(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q < 0:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_115(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 1:
        raise ValueError(
            f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    return p, q, m


def x_estimate_bass_mom__mutmut_116(
    t: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float, float]:
    """Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
    ----
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns
    -------
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError(
            "Input sequences t and y must have the same length and at least 3 data points.",
        )

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame(
        {"x_t": incremental_adoptions, "y_t_minus_1": lagged_cumulative_adoptions},
    )

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError(
            "Not enough valid data points for Bass MoM estimation after preprocessing.",
        )

    X = pd.DataFrame(
        {
            "intercept": 1,
            "y_t_minus_1": data_for_reg["y_t_minus_1"],
            "y_t_minus_1_sq": data_for_reg["y_t_minus_1"] ** 2,
        },
    )
    y_reg = data_for_reg["x_t"]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.",
        )

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.",
        )

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError(
            "No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.",
        )

    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(
            None,
        )

    return p, q, m

x_estimate_bass_mom__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_estimate_bass_mom__mutmut_1': x_estimate_bass_mom__mutmut_1, 
    'x_estimate_bass_mom__mutmut_2': x_estimate_bass_mom__mutmut_2, 
    'x_estimate_bass_mom__mutmut_3': x_estimate_bass_mom__mutmut_3, 
    'x_estimate_bass_mom__mutmut_4': x_estimate_bass_mom__mutmut_4, 
    'x_estimate_bass_mom__mutmut_5': x_estimate_bass_mom__mutmut_5, 
    'x_estimate_bass_mom__mutmut_6': x_estimate_bass_mom__mutmut_6, 
    'x_estimate_bass_mom__mutmut_7': x_estimate_bass_mom__mutmut_7, 
    'x_estimate_bass_mom__mutmut_8': x_estimate_bass_mom__mutmut_8, 
    'x_estimate_bass_mom__mutmut_9': x_estimate_bass_mom__mutmut_9, 
    'x_estimate_bass_mom__mutmut_10': x_estimate_bass_mom__mutmut_10, 
    'x_estimate_bass_mom__mutmut_11': x_estimate_bass_mom__mutmut_11, 
    'x_estimate_bass_mom__mutmut_12': x_estimate_bass_mom__mutmut_12, 
    'x_estimate_bass_mom__mutmut_13': x_estimate_bass_mom__mutmut_13, 
    'x_estimate_bass_mom__mutmut_14': x_estimate_bass_mom__mutmut_14, 
    'x_estimate_bass_mom__mutmut_15': x_estimate_bass_mom__mutmut_15, 
    'x_estimate_bass_mom__mutmut_16': x_estimate_bass_mom__mutmut_16, 
    'x_estimate_bass_mom__mutmut_17': x_estimate_bass_mom__mutmut_17, 
    'x_estimate_bass_mom__mutmut_18': x_estimate_bass_mom__mutmut_18, 
    'x_estimate_bass_mom__mutmut_19': x_estimate_bass_mom__mutmut_19, 
    'x_estimate_bass_mom__mutmut_20': x_estimate_bass_mom__mutmut_20, 
    'x_estimate_bass_mom__mutmut_21': x_estimate_bass_mom__mutmut_21, 
    'x_estimate_bass_mom__mutmut_22': x_estimate_bass_mom__mutmut_22, 
    'x_estimate_bass_mom__mutmut_23': x_estimate_bass_mom__mutmut_23, 
    'x_estimate_bass_mom__mutmut_24': x_estimate_bass_mom__mutmut_24, 
    'x_estimate_bass_mom__mutmut_25': x_estimate_bass_mom__mutmut_25, 
    'x_estimate_bass_mom__mutmut_26': x_estimate_bass_mom__mutmut_26, 
    'x_estimate_bass_mom__mutmut_27': x_estimate_bass_mom__mutmut_27, 
    'x_estimate_bass_mom__mutmut_28': x_estimate_bass_mom__mutmut_28, 
    'x_estimate_bass_mom__mutmut_29': x_estimate_bass_mom__mutmut_29, 
    'x_estimate_bass_mom__mutmut_30': x_estimate_bass_mom__mutmut_30, 
    'x_estimate_bass_mom__mutmut_31': x_estimate_bass_mom__mutmut_31, 
    'x_estimate_bass_mom__mutmut_32': x_estimate_bass_mom__mutmut_32, 
    'x_estimate_bass_mom__mutmut_33': x_estimate_bass_mom__mutmut_33, 
    'x_estimate_bass_mom__mutmut_34': x_estimate_bass_mom__mutmut_34, 
    'x_estimate_bass_mom__mutmut_35': x_estimate_bass_mom__mutmut_35, 
    'x_estimate_bass_mom__mutmut_36': x_estimate_bass_mom__mutmut_36, 
    'x_estimate_bass_mom__mutmut_37': x_estimate_bass_mom__mutmut_37, 
    'x_estimate_bass_mom__mutmut_38': x_estimate_bass_mom__mutmut_38, 
    'x_estimate_bass_mom__mutmut_39': x_estimate_bass_mom__mutmut_39, 
    'x_estimate_bass_mom__mutmut_40': x_estimate_bass_mom__mutmut_40, 
    'x_estimate_bass_mom__mutmut_41': x_estimate_bass_mom__mutmut_41, 
    'x_estimate_bass_mom__mutmut_42': x_estimate_bass_mom__mutmut_42, 
    'x_estimate_bass_mom__mutmut_43': x_estimate_bass_mom__mutmut_43, 
    'x_estimate_bass_mom__mutmut_44': x_estimate_bass_mom__mutmut_44, 
    'x_estimate_bass_mom__mutmut_45': x_estimate_bass_mom__mutmut_45, 
    'x_estimate_bass_mom__mutmut_46': x_estimate_bass_mom__mutmut_46, 
    'x_estimate_bass_mom__mutmut_47': x_estimate_bass_mom__mutmut_47, 
    'x_estimate_bass_mom__mutmut_48': x_estimate_bass_mom__mutmut_48, 
    'x_estimate_bass_mom__mutmut_49': x_estimate_bass_mom__mutmut_49, 
    'x_estimate_bass_mom__mutmut_50': x_estimate_bass_mom__mutmut_50, 
    'x_estimate_bass_mom__mutmut_51': x_estimate_bass_mom__mutmut_51, 
    'x_estimate_bass_mom__mutmut_52': x_estimate_bass_mom__mutmut_52, 
    'x_estimate_bass_mom__mutmut_53': x_estimate_bass_mom__mutmut_53, 
    'x_estimate_bass_mom__mutmut_54': x_estimate_bass_mom__mutmut_54, 
    'x_estimate_bass_mom__mutmut_55': x_estimate_bass_mom__mutmut_55, 
    'x_estimate_bass_mom__mutmut_56': x_estimate_bass_mom__mutmut_56, 
    'x_estimate_bass_mom__mutmut_57': x_estimate_bass_mom__mutmut_57, 
    'x_estimate_bass_mom__mutmut_58': x_estimate_bass_mom__mutmut_58, 
    'x_estimate_bass_mom__mutmut_59': x_estimate_bass_mom__mutmut_59, 
    'x_estimate_bass_mom__mutmut_60': x_estimate_bass_mom__mutmut_60, 
    'x_estimate_bass_mom__mutmut_61': x_estimate_bass_mom__mutmut_61, 
    'x_estimate_bass_mom__mutmut_62': x_estimate_bass_mom__mutmut_62, 
    'x_estimate_bass_mom__mutmut_63': x_estimate_bass_mom__mutmut_63, 
    'x_estimate_bass_mom__mutmut_64': x_estimate_bass_mom__mutmut_64, 
    'x_estimate_bass_mom__mutmut_65': x_estimate_bass_mom__mutmut_65, 
    'x_estimate_bass_mom__mutmut_66': x_estimate_bass_mom__mutmut_66, 
    'x_estimate_bass_mom__mutmut_67': x_estimate_bass_mom__mutmut_67, 
    'x_estimate_bass_mom__mutmut_68': x_estimate_bass_mom__mutmut_68, 
    'x_estimate_bass_mom__mutmut_69': x_estimate_bass_mom__mutmut_69, 
    'x_estimate_bass_mom__mutmut_70': x_estimate_bass_mom__mutmut_70, 
    'x_estimate_bass_mom__mutmut_71': x_estimate_bass_mom__mutmut_71, 
    'x_estimate_bass_mom__mutmut_72': x_estimate_bass_mom__mutmut_72, 
    'x_estimate_bass_mom__mutmut_73': x_estimate_bass_mom__mutmut_73, 
    'x_estimate_bass_mom__mutmut_74': x_estimate_bass_mom__mutmut_74, 
    'x_estimate_bass_mom__mutmut_75': x_estimate_bass_mom__mutmut_75, 
    'x_estimate_bass_mom__mutmut_76': x_estimate_bass_mom__mutmut_76, 
    'x_estimate_bass_mom__mutmut_77': x_estimate_bass_mom__mutmut_77, 
    'x_estimate_bass_mom__mutmut_78': x_estimate_bass_mom__mutmut_78, 
    'x_estimate_bass_mom__mutmut_79': x_estimate_bass_mom__mutmut_79, 
    'x_estimate_bass_mom__mutmut_80': x_estimate_bass_mom__mutmut_80, 
    'x_estimate_bass_mom__mutmut_81': x_estimate_bass_mom__mutmut_81, 
    'x_estimate_bass_mom__mutmut_82': x_estimate_bass_mom__mutmut_82, 
    'x_estimate_bass_mom__mutmut_83': x_estimate_bass_mom__mutmut_83, 
    'x_estimate_bass_mom__mutmut_84': x_estimate_bass_mom__mutmut_84, 
    'x_estimate_bass_mom__mutmut_85': x_estimate_bass_mom__mutmut_85, 
    'x_estimate_bass_mom__mutmut_86': x_estimate_bass_mom__mutmut_86, 
    'x_estimate_bass_mom__mutmut_87': x_estimate_bass_mom__mutmut_87, 
    'x_estimate_bass_mom__mutmut_88': x_estimate_bass_mom__mutmut_88, 
    'x_estimate_bass_mom__mutmut_89': x_estimate_bass_mom__mutmut_89, 
    'x_estimate_bass_mom__mutmut_90': x_estimate_bass_mom__mutmut_90, 
    'x_estimate_bass_mom__mutmut_91': x_estimate_bass_mom__mutmut_91, 
    'x_estimate_bass_mom__mutmut_92': x_estimate_bass_mom__mutmut_92, 
    'x_estimate_bass_mom__mutmut_93': x_estimate_bass_mom__mutmut_93, 
    'x_estimate_bass_mom__mutmut_94': x_estimate_bass_mom__mutmut_94, 
    'x_estimate_bass_mom__mutmut_95': x_estimate_bass_mom__mutmut_95, 
    'x_estimate_bass_mom__mutmut_96': x_estimate_bass_mom__mutmut_96, 
    'x_estimate_bass_mom__mutmut_97': x_estimate_bass_mom__mutmut_97, 
    'x_estimate_bass_mom__mutmut_98': x_estimate_bass_mom__mutmut_98, 
    'x_estimate_bass_mom__mutmut_99': x_estimate_bass_mom__mutmut_99, 
    'x_estimate_bass_mom__mutmut_100': x_estimate_bass_mom__mutmut_100, 
    'x_estimate_bass_mom__mutmut_101': x_estimate_bass_mom__mutmut_101, 
    'x_estimate_bass_mom__mutmut_102': x_estimate_bass_mom__mutmut_102, 
    'x_estimate_bass_mom__mutmut_103': x_estimate_bass_mom__mutmut_103, 
    'x_estimate_bass_mom__mutmut_104': x_estimate_bass_mom__mutmut_104, 
    'x_estimate_bass_mom__mutmut_105': x_estimate_bass_mom__mutmut_105, 
    'x_estimate_bass_mom__mutmut_106': x_estimate_bass_mom__mutmut_106, 
    'x_estimate_bass_mom__mutmut_107': x_estimate_bass_mom__mutmut_107, 
    'x_estimate_bass_mom__mutmut_108': x_estimate_bass_mom__mutmut_108, 
    'x_estimate_bass_mom__mutmut_109': x_estimate_bass_mom__mutmut_109, 
    'x_estimate_bass_mom__mutmut_110': x_estimate_bass_mom__mutmut_110, 
    'x_estimate_bass_mom__mutmut_111': x_estimate_bass_mom__mutmut_111, 
    'x_estimate_bass_mom__mutmut_112': x_estimate_bass_mom__mutmut_112, 
    'x_estimate_bass_mom__mutmut_113': x_estimate_bass_mom__mutmut_113, 
    'x_estimate_bass_mom__mutmut_114': x_estimate_bass_mom__mutmut_114, 
    'x_estimate_bass_mom__mutmut_115': x_estimate_bass_mom__mutmut_115, 
    'x_estimate_bass_mom__mutmut_116': x_estimate_bass_mom__mutmut_116
}
x_estimate_bass_mom__mutmut_orig.__name__ = 'x_estimate_bass_mom'


class MoMFitter:
    """Fitter for the Bass Diffusion Model using the Method of Moments (MoM).
    This fitter is specifically designed for the BassModel.
    """

    def __init__(self):
        args = []# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMoMFitterǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁMoMFitterǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁMoMFitterǁ__init____mutmut_orig(self):
        self._params: dict[str, float] = {}

    def xǁMoMFitterǁ__init____mutmut_1(self):
        self._params: dict[str, float] = None
    
    xǁMoMFitterǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMoMFitterǁ__init____mutmut_1': xǁMoMFitterǁ__init____mutmut_1
    }
    xǁMoMFitterǁ__init____mutmut_orig.__name__ = 'xǁMoMFitterǁ__init__'

    def fit(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        args = [model, t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMoMFitterǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁMoMFitterǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁMoMFitterǁfit__mutmut_orig(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_1(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_2(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError(None)

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_3(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("XXMoMFitter can only fit BassModel instances.XX")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_4(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("momfitter can only fit bassmodel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_5(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MOMFITTER CAN ONLY FIT BASSMODEL INSTANCES.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_6(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = None
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_7(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(None, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_8(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, None)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_9(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_10(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, )
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_11(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = None
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_12(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"XXpXX": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_13(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"P": p, "q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_14(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "XXqXX": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_15(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "Q": q, "m": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_16(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "XXmXX": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_17(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "M": m}
        self._params = model.params_  # Store fitted parameters internally
        return model

    def xǁMoMFitterǁfit__mutmut_18(
        self,
        model: DiffusionModel,
        t: Sequence[float],
        y: Sequence[float],
    ) -> DiffusionModel:
        """Fits the BassModel using the Method of Moments.

        Args:
        ----
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns
        -------
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import (
            BassModel,
        )

        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = None  # Store fitted parameters internally
        return model
    
    xǁMoMFitterǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMoMFitterǁfit__mutmut_1': xǁMoMFitterǁfit__mutmut_1, 
        'xǁMoMFitterǁfit__mutmut_2': xǁMoMFitterǁfit__mutmut_2, 
        'xǁMoMFitterǁfit__mutmut_3': xǁMoMFitterǁfit__mutmut_3, 
        'xǁMoMFitterǁfit__mutmut_4': xǁMoMFitterǁfit__mutmut_4, 
        'xǁMoMFitterǁfit__mutmut_5': xǁMoMFitterǁfit__mutmut_5, 
        'xǁMoMFitterǁfit__mutmut_6': xǁMoMFitterǁfit__mutmut_6, 
        'xǁMoMFitterǁfit__mutmut_7': xǁMoMFitterǁfit__mutmut_7, 
        'xǁMoMFitterǁfit__mutmut_8': xǁMoMFitterǁfit__mutmut_8, 
        'xǁMoMFitterǁfit__mutmut_9': xǁMoMFitterǁfit__mutmut_9, 
        'xǁMoMFitterǁfit__mutmut_10': xǁMoMFitterǁfit__mutmut_10, 
        'xǁMoMFitterǁfit__mutmut_11': xǁMoMFitterǁfit__mutmut_11, 
        'xǁMoMFitterǁfit__mutmut_12': xǁMoMFitterǁfit__mutmut_12, 
        'xǁMoMFitterǁfit__mutmut_13': xǁMoMFitterǁfit__mutmut_13, 
        'xǁMoMFitterǁfit__mutmut_14': xǁMoMFitterǁfit__mutmut_14, 
        'xǁMoMFitterǁfit__mutmut_15': xǁMoMFitterǁfit__mutmut_15, 
        'xǁMoMFitterǁfit__mutmut_16': xǁMoMFitterǁfit__mutmut_16, 
        'xǁMoMFitterǁfit__mutmut_17': xǁMoMFitterǁfit__mutmut_17, 
        'xǁMoMFitterǁfit__mutmut_18': xǁMoMFitterǁfit__mutmut_18
    }
    xǁMoMFitterǁfit__mutmut_orig.__name__ = 'xǁMoMFitterǁfit'

    @property
    def params_(self) -> dict[str, float]:
        return self._params
