"""Validation utilities for the Innovate library."""

import numbers
from collections.abc import Sequence

import numpy as np


def validate_sequence_numeric(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}") from e

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Parameter '{param_name}' must contain numeric values")

    return arr


def validate_positive_numeric_sequence(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def validate_float(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except TypeError, ValueError:
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def validate_probability(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=0.0, max_val=1.0)


def validate_covariates(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def validate_time_series(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) != len(y_arr):
        raise ValueError(
            f"Length of '{param_name_t}' ({len(t_arr)}) must match length of '{param_name_y}' ({len(y_arr)})"
        )

    if len(t_arr) < 2:
        raise ValueError(f"'{param_name_t}' and '{param_name_y}' must have at least 2 points for fitting")

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def validate_covariates_dict(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("Covariates must be a dictionary or None")

    result = {}
    for cov_name, cov_values in covariates_dict.items():
        if not isinstance(cov_name, str):
            raise TypeError(f"Covariate names must be strings, got {type(cov_name)} for key")

        if cov_name not in expected_covariates:
            raise ValueError(f"Unknown covariate '{cov_name}', expected one of: {expected_covariates}")

        cov_arr = validate_sequence_numeric(cov_values, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result
