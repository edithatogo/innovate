"""Validation utilities for the Innovate library."""

import numbers
from collections.abc import Sequence

import numpy as np
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


def validate_sequence_numeric(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    args = [sequence, param_name, allow_empty]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_sequence_numeric__mutmut_orig, x_validate_sequence_numeric__mutmut_mutants, args, kwargs, None)


def x_validate_sequence_numeric__mutmut_orig(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_1(sequence: Sequence, param_name: str, allow_empty: bool = True) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_2(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is not None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_3(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(None)

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_4(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") and isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_5(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_6(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(None, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_7(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, None) or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_8(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr("__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_9(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, ) or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_10(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "XX__iter__XX") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_11(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__ITER__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_12(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(None)

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_13(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(None)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_14(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty or len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_15(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_16(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) != 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_17(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 1:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_18(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(None)

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(sequence)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_19(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = None
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_20(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate that a sequence contains numeric values."""
    if sequence is None:
        raise ValueError(f"Parameter '{param_name}' cannot be None")

    if not hasattr(sequence, "__iter__") or isinstance(sequence, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence, got {type(sequence)}")

    if not allow_empty and len(sequence) == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be empty")

    # Convert to numpy array to validate numeric content
    try:
        arr = np.asarray(None)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_21(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_22(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(None, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_23(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, None):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_24(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_25(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, ):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_26(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(None)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Parameter '{param_name}' values must be numeric: {e}")

    return arr


def x_validate_sequence_numeric__mutmut_27(sequence: Sequence, param_name: str, allow_empty: bool = False) -> np.ndarray:
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
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"Parameter '{param_name}' must contain numeric values")
    except (TypeError, ValueError) as e:
        raise TypeError(None)

    return arr

x_validate_sequence_numeric__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_sequence_numeric__mutmut_1': x_validate_sequence_numeric__mutmut_1, 
    'x_validate_sequence_numeric__mutmut_2': x_validate_sequence_numeric__mutmut_2, 
    'x_validate_sequence_numeric__mutmut_3': x_validate_sequence_numeric__mutmut_3, 
    'x_validate_sequence_numeric__mutmut_4': x_validate_sequence_numeric__mutmut_4, 
    'x_validate_sequence_numeric__mutmut_5': x_validate_sequence_numeric__mutmut_5, 
    'x_validate_sequence_numeric__mutmut_6': x_validate_sequence_numeric__mutmut_6, 
    'x_validate_sequence_numeric__mutmut_7': x_validate_sequence_numeric__mutmut_7, 
    'x_validate_sequence_numeric__mutmut_8': x_validate_sequence_numeric__mutmut_8, 
    'x_validate_sequence_numeric__mutmut_9': x_validate_sequence_numeric__mutmut_9, 
    'x_validate_sequence_numeric__mutmut_10': x_validate_sequence_numeric__mutmut_10, 
    'x_validate_sequence_numeric__mutmut_11': x_validate_sequence_numeric__mutmut_11, 
    'x_validate_sequence_numeric__mutmut_12': x_validate_sequence_numeric__mutmut_12, 
    'x_validate_sequence_numeric__mutmut_13': x_validate_sequence_numeric__mutmut_13, 
    'x_validate_sequence_numeric__mutmut_14': x_validate_sequence_numeric__mutmut_14, 
    'x_validate_sequence_numeric__mutmut_15': x_validate_sequence_numeric__mutmut_15, 
    'x_validate_sequence_numeric__mutmut_16': x_validate_sequence_numeric__mutmut_16, 
    'x_validate_sequence_numeric__mutmut_17': x_validate_sequence_numeric__mutmut_17, 
    'x_validate_sequence_numeric__mutmut_18': x_validate_sequence_numeric__mutmut_18, 
    'x_validate_sequence_numeric__mutmut_19': x_validate_sequence_numeric__mutmut_19, 
    'x_validate_sequence_numeric__mutmut_20': x_validate_sequence_numeric__mutmut_20, 
    'x_validate_sequence_numeric__mutmut_21': x_validate_sequence_numeric__mutmut_21, 
    'x_validate_sequence_numeric__mutmut_22': x_validate_sequence_numeric__mutmut_22, 
    'x_validate_sequence_numeric__mutmut_23': x_validate_sequence_numeric__mutmut_23, 
    'x_validate_sequence_numeric__mutmut_24': x_validate_sequence_numeric__mutmut_24, 
    'x_validate_sequence_numeric__mutmut_25': x_validate_sequence_numeric__mutmut_25, 
    'x_validate_sequence_numeric__mutmut_26': x_validate_sequence_numeric__mutmut_26, 
    'x_validate_sequence_numeric__mutmut_27': x_validate_sequence_numeric__mutmut_27
}
x_validate_sequence_numeric__mutmut_orig.__name__ = 'x_validate_sequence_numeric'


def validate_positive_numeric_sequence(sequence: Sequence, param_name: str) -> np.ndarray:
    args = [sequence, param_name]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_positive_numeric_sequence__mutmut_orig, x_validate_positive_numeric_sequence__mutmut_mutants, args, kwargs, None)


def x_validate_positive_numeric_sequence__mutmut_orig(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_1(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = None

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_2(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(None, param_name)

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_3(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, None)

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_4(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(param_name)

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_5(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, )

    if np.any(arr < 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_6(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(None):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_7(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(arr <= 0):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_8(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(arr < 1):
        raise ValueError(f"Parameter '{param_name}' must contain non-negative values")

    return arr


def x_validate_positive_numeric_sequence__mutmut_9(sequence: Sequence, param_name: str) -> np.ndarray:
    """Validate that a sequence contains positive numeric values."""
    arr = validate_sequence_numeric(sequence, param_name)

    if np.any(arr < 0):
        raise ValueError(None)

    return arr

x_validate_positive_numeric_sequence__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_positive_numeric_sequence__mutmut_1': x_validate_positive_numeric_sequence__mutmut_1, 
    'x_validate_positive_numeric_sequence__mutmut_2': x_validate_positive_numeric_sequence__mutmut_2, 
    'x_validate_positive_numeric_sequence__mutmut_3': x_validate_positive_numeric_sequence__mutmut_3, 
    'x_validate_positive_numeric_sequence__mutmut_4': x_validate_positive_numeric_sequence__mutmut_4, 
    'x_validate_positive_numeric_sequence__mutmut_5': x_validate_positive_numeric_sequence__mutmut_5, 
    'x_validate_positive_numeric_sequence__mutmut_6': x_validate_positive_numeric_sequence__mutmut_6, 
    'x_validate_positive_numeric_sequence__mutmut_7': x_validate_positive_numeric_sequence__mutmut_7, 
    'x_validate_positive_numeric_sequence__mutmut_8': x_validate_positive_numeric_sequence__mutmut_8, 
    'x_validate_positive_numeric_sequence__mutmut_9': x_validate_positive_numeric_sequence__mutmut_9
}
x_validate_positive_numeric_sequence__mutmut_orig.__name__ = 'x_validate_positive_numeric_sequence'


def validate_float(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    args = [value, param_name, min_val, max_val]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_float__mutmut_orig, x_validate_float__mutmut_mutants, args, kwargs, None)


def x_validate_float__mutmut_orig(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_1(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_2(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(None)

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_3(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(None)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_4(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = None
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_5(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(None)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_6(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(None)

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_7(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None or float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_8(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_9(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val <= min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_10(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(None)

    if max_val is not None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_11(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None or float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_12(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is None and float_val > max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_13(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val >= max_val:
        raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {float_val}")

    return float_val


def x_validate_float__mutmut_14(
    value: float | int, param_name: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Validate that a value is a float within optional bounds."""
    if not isinstance(value, (numbers.Real, np.number)):
        raise TypeError(f"Parameter '{param_name}' must be numeric, got {type(value)}")

    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Parameter '{param_name}' must be convertible to float, got {value}")

    if min_val is not None and float_val < min_val:
        raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {float_val}")

    if max_val is not None and float_val > max_val:
        raise ValueError(None)

    return float_val

x_validate_float__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_float__mutmut_1': x_validate_float__mutmut_1, 
    'x_validate_float__mutmut_2': x_validate_float__mutmut_2, 
    'x_validate_float__mutmut_3': x_validate_float__mutmut_3, 
    'x_validate_float__mutmut_4': x_validate_float__mutmut_4, 
    'x_validate_float__mutmut_5': x_validate_float__mutmut_5, 
    'x_validate_float__mutmut_6': x_validate_float__mutmut_6, 
    'x_validate_float__mutmut_7': x_validate_float__mutmut_7, 
    'x_validate_float__mutmut_8': x_validate_float__mutmut_8, 
    'x_validate_float__mutmut_9': x_validate_float__mutmut_9, 
    'x_validate_float__mutmut_10': x_validate_float__mutmut_10, 
    'x_validate_float__mutmut_11': x_validate_float__mutmut_11, 
    'x_validate_float__mutmut_12': x_validate_float__mutmut_12, 
    'x_validate_float__mutmut_13': x_validate_float__mutmut_13, 
    'x_validate_float__mutmut_14': x_validate_float__mutmut_14
}
x_validate_float__mutmut_orig.__name__ = 'x_validate_float'


def validate_probability(value: float | int, param_name: str) -> float:
    args = [value, param_name]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_probability__mutmut_orig, x_validate_probability__mutmut_mutants, args, kwargs, None)


def x_validate_probability__mutmut_orig(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=0.0, max_val=1.0)


def x_validate_probability__mutmut_1(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(None, param_name, min_val=0.0, max_val=1.0)


def x_validate_probability__mutmut_2(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, None, min_val=0.0, max_val=1.0)


def x_validate_probability__mutmut_3(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=None, max_val=1.0)


def x_validate_probability__mutmut_4(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=0.0, max_val=None)


def x_validate_probability__mutmut_5(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(param_name, min_val=0.0, max_val=1.0)


def x_validate_probability__mutmut_6(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, min_val=0.0, max_val=1.0)


def x_validate_probability__mutmut_7(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, max_val=1.0)


def x_validate_probability__mutmut_8(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=0.0, )


def x_validate_probability__mutmut_9(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=1.0, max_val=1.0)


def x_validate_probability__mutmut_10(value: float | int, param_name: str) -> float:
    """Validate that a value is a probability (between 0 and 1)."""
    return validate_float(value, param_name, min_val=0.0, max_val=2.0)

x_validate_probability__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_probability__mutmut_1': x_validate_probability__mutmut_1, 
    'x_validate_probability__mutmut_2': x_validate_probability__mutmut_2, 
    'x_validate_probability__mutmut_3': x_validate_probability__mutmut_3, 
    'x_validate_probability__mutmut_4': x_validate_probability__mutmut_4, 
    'x_validate_probability__mutmut_5': x_validate_probability__mutmut_5, 
    'x_validate_probability__mutmut_6': x_validate_probability__mutmut_6, 
    'x_validate_probability__mutmut_7': x_validate_probability__mutmut_7, 
    'x_validate_probability__mutmut_8': x_validate_probability__mutmut_8, 
    'x_validate_probability__mutmut_9': x_validate_probability__mutmut_9, 
    'x_validate_probability__mutmut_10': x_validate_probability__mutmut_10
}
x_validate_probability__mutmut_orig.__name__ = 'x_validate_probability'


def validate_covariates(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    args = [covariates, param_name]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_covariates__mutmut_orig, x_validate_covariates__mutmut_mutants, args, kwargs, None)


def x_validate_covariates__mutmut_orig(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
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


def x_validate_covariates__mutmut_1(covariates: Sequence[str] | None, param_name: str = "XXcovariatesXX") -> Sequence[str]:
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


def x_validate_covariates__mutmut_2(covariates: Sequence[str] | None, param_name: str = "COVARIATES") -> Sequence[str]:
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


def x_validate_covariates__mutmut_3(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is not None:
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


def x_validate_covariates__mutmut_4(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(None)

    if not hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_5(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_6(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(None, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_7(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, None):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_8(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr("__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_9(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, ):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_10(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "XX__iter__XX"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_11(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__ITER__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_12(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__iter__"):
        raise TypeError(None)

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_13(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(None)}")

    result = []
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_14(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = None
    for i, cov in enumerate(covariates):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_15(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
    """Validate covariates parameter."""
    if covariates is None:
        return []

    # Reject strings specifically since they're iterable but not what we want
    if isinstance(covariates, str):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, not a string")

    if not hasattr(covariates, "__iter__"):
        raise TypeError(f"Parameter '{param_name}' must be a sequence of strings, got {type(covariates)}")

    result = []
    for i, cov in enumerate(None):
        if not isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_16(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
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
        if isinstance(cov, str):
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(cov)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_17(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
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
            raise TypeError(None)
        result.append(cov)

    return result


def x_validate_covariates__mutmut_18(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
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
            raise TypeError(f"Element {i} of '{param_name}' must be a string, got {type(None)}")
        result.append(cov)

    return result


def x_validate_covariates__mutmut_19(covariates: Sequence[str] | None, param_name: str = "covariates") -> Sequence[str]:
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
        result.append(None)

    return result

x_validate_covariates__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_covariates__mutmut_1': x_validate_covariates__mutmut_1, 
    'x_validate_covariates__mutmut_2': x_validate_covariates__mutmut_2, 
    'x_validate_covariates__mutmut_3': x_validate_covariates__mutmut_3, 
    'x_validate_covariates__mutmut_4': x_validate_covariates__mutmut_4, 
    'x_validate_covariates__mutmut_5': x_validate_covariates__mutmut_5, 
    'x_validate_covariates__mutmut_6': x_validate_covariates__mutmut_6, 
    'x_validate_covariates__mutmut_7': x_validate_covariates__mutmut_7, 
    'x_validate_covariates__mutmut_8': x_validate_covariates__mutmut_8, 
    'x_validate_covariates__mutmut_9': x_validate_covariates__mutmut_9, 
    'x_validate_covariates__mutmut_10': x_validate_covariates__mutmut_10, 
    'x_validate_covariates__mutmut_11': x_validate_covariates__mutmut_11, 
    'x_validate_covariates__mutmut_12': x_validate_covariates__mutmut_12, 
    'x_validate_covariates__mutmut_13': x_validate_covariates__mutmut_13, 
    'x_validate_covariates__mutmut_14': x_validate_covariates__mutmut_14, 
    'x_validate_covariates__mutmut_15': x_validate_covariates__mutmut_15, 
    'x_validate_covariates__mutmut_16': x_validate_covariates__mutmut_16, 
    'x_validate_covariates__mutmut_17': x_validate_covariates__mutmut_17, 
    'x_validate_covariates__mutmut_18': x_validate_covariates__mutmut_18, 
    'x_validate_covariates__mutmut_19': x_validate_covariates__mutmut_19
}
x_validate_covariates__mutmut_orig.__name__ = 'x_validate_covariates'


def validate_time_series(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    args = [t, y, param_name_t, param_name_y]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_time_series__mutmut_orig, x_validate_time_series__mutmut_mutants, args, kwargs, None)


def x_validate_time_series__mutmut_orig(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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


def x_validate_time_series__mutmut_1(t: Sequence, y: Sequence, param_name_t: str = "XXtXX", param_name_y: str = "y") -> tuple:
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


def x_validate_time_series__mutmut_2(t: Sequence, y: Sequence, param_name_t: str = "T", param_name_y: str = "y") -> tuple:
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


def x_validate_time_series__mutmut_3(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "XXyXX") -> tuple:
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


def x_validate_time_series__mutmut_4(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "Y") -> tuple:
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


def x_validate_time_series__mutmut_5(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = None
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


def x_validate_time_series__mutmut_6(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(None, param_name_t)
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


def x_validate_time_series__mutmut_7(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, None)
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


def x_validate_time_series__mutmut_8(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(param_name_t)
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


def x_validate_time_series__mutmut_9(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, )
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


def x_validate_time_series__mutmut_10(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = None

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


def x_validate_time_series__mutmut_11(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(None, param_name_y)

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


def x_validate_time_series__mutmut_12(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, None)

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


def x_validate_time_series__mutmut_13(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(param_name_y)

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


def x_validate_time_series__mutmut_14(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, )

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


def x_validate_time_series__mutmut_15(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) == len(y_arr):
        raise ValueError(
            f"Length of '{param_name_t}' ({len(t_arr)}) must match length of '{param_name_y}' ({len(y_arr)})"
        )

    if len(t_arr) < 2:
        raise ValueError(f"'{param_name_t}' and '{param_name_y}' must have at least 2 points for fitting")

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_16(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) != len(y_arr):
        raise ValueError(
            None
        )

    if len(t_arr) < 2:
        raise ValueError(f"'{param_name_t}' and '{param_name_y}' must have at least 2 points for fitting")

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_17(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) != len(y_arr):
        raise ValueError(
            f"Length of '{param_name_t}' ({len(t_arr)}) must match length of '{param_name_y}' ({len(y_arr)})"
        )

    if len(t_arr) <= 2:
        raise ValueError(f"'{param_name_t}' and '{param_name_y}' must have at least 2 points for fitting")

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_18(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) != len(y_arr):
        raise ValueError(
            f"Length of '{param_name_t}' ({len(t_arr)}) must match length of '{param_name_y}' ({len(y_arr)})"
        )

    if len(t_arr) < 3:
        raise ValueError(f"'{param_name_t}' and '{param_name_y}' must have at least 2 points for fitting")

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_19(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
    """Validate time series data for fitting."""
    t_arr = validate_sequence_numeric(t, param_name_t)
    y_arr = validate_positive_numeric_sequence(y, param_name_y)

    if len(t_arr) != len(y_arr):
        raise ValueError(
            f"Length of '{param_name_t}' ({len(t_arr)}) must match length of '{param_name_y}' ({len(y_arr)})"
        )

    if len(t_arr) < 2:
        raise ValueError(None)

    # Check for non-decreasing time (allowing for equal values)
    if not np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_20(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
    if np.all(np.diff(t_arr) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_21(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
    if not np.all(None):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_22(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
    if not np.all(np.diff(None) >= 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_23(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
    if not np.all(np.diff(t_arr) > 0):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_24(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
    if not np.all(np.diff(t_arr) >= 1):
        raise ValueError(f"'{param_name_t}' values must be non-decreasing")

    return t_arr, y_arr


def x_validate_time_series__mutmut_25(t: Sequence, y: Sequence, param_name_t: str = "t", param_name_y: str = "y") -> tuple:
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
        raise ValueError(None)

    return t_arr, y_arr

x_validate_time_series__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_time_series__mutmut_1': x_validate_time_series__mutmut_1, 
    'x_validate_time_series__mutmut_2': x_validate_time_series__mutmut_2, 
    'x_validate_time_series__mutmut_3': x_validate_time_series__mutmut_3, 
    'x_validate_time_series__mutmut_4': x_validate_time_series__mutmut_4, 
    'x_validate_time_series__mutmut_5': x_validate_time_series__mutmut_5, 
    'x_validate_time_series__mutmut_6': x_validate_time_series__mutmut_6, 
    'x_validate_time_series__mutmut_7': x_validate_time_series__mutmut_7, 
    'x_validate_time_series__mutmut_8': x_validate_time_series__mutmut_8, 
    'x_validate_time_series__mutmut_9': x_validate_time_series__mutmut_9, 
    'x_validate_time_series__mutmut_10': x_validate_time_series__mutmut_10, 
    'x_validate_time_series__mutmut_11': x_validate_time_series__mutmut_11, 
    'x_validate_time_series__mutmut_12': x_validate_time_series__mutmut_12, 
    'x_validate_time_series__mutmut_13': x_validate_time_series__mutmut_13, 
    'x_validate_time_series__mutmut_14': x_validate_time_series__mutmut_14, 
    'x_validate_time_series__mutmut_15': x_validate_time_series__mutmut_15, 
    'x_validate_time_series__mutmut_16': x_validate_time_series__mutmut_16, 
    'x_validate_time_series__mutmut_17': x_validate_time_series__mutmut_17, 
    'x_validate_time_series__mutmut_18': x_validate_time_series__mutmut_18, 
    'x_validate_time_series__mutmut_19': x_validate_time_series__mutmut_19, 
    'x_validate_time_series__mutmut_20': x_validate_time_series__mutmut_20, 
    'x_validate_time_series__mutmut_21': x_validate_time_series__mutmut_21, 
    'x_validate_time_series__mutmut_22': x_validate_time_series__mutmut_22, 
    'x_validate_time_series__mutmut_23': x_validate_time_series__mutmut_23, 
    'x_validate_time_series__mutmut_24': x_validate_time_series__mutmut_24, 
    'x_validate_time_series__mutmut_25': x_validate_time_series__mutmut_25
}
x_validate_time_series__mutmut_orig.__name__ = 'x_validate_time_series'


def validate_covariates_dict(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    args = [covariates_dict, expected_covariates, t_length]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_validate_covariates_dict__mutmut_orig, x_validate_covariates_dict__mutmut_mutants, args, kwargs, None)


def x_validate_covariates_dict__mutmut_orig(
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


def x_validate_covariates_dict__mutmut_1(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is not None:
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


def x_validate_covariates_dict__mutmut_2(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if isinstance(covariates_dict, dict):
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


def x_validate_covariates_dict__mutmut_3(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError(None)

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


def x_validate_covariates_dict__mutmut_4(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("XXCovariates must be a dictionary or NoneXX")

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


def x_validate_covariates_dict__mutmut_5(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("covariates must be a dictionary or none")

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


def x_validate_covariates_dict__mutmut_6(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("COVARIATES MUST BE A DICTIONARY OR NONE")

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


def x_validate_covariates_dict__mutmut_7(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("Covariates must be a dictionary or None")

    result = None
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


def x_validate_covariates_dict__mutmut_8(
    covariates_dict: dict[str, Sequence] | None, expected_covariates: Sequence[str], t_length: int
) -> dict[str, np.ndarray] | None:
    """Validate covariates dictionary."""
    if covariates_dict is None:
        return None

    if not isinstance(covariates_dict, dict):
        raise TypeError("Covariates must be a dictionary or None")

    result = {}
    for cov_name, cov_values in covariates_dict.items():
        if isinstance(cov_name, str):
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


def x_validate_covariates_dict__mutmut_9(
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
            raise TypeError(None)

        if cov_name not in expected_covariates:
            raise ValueError(f"Unknown covariate '{cov_name}', expected one of: {expected_covariates}")

        cov_arr = validate_sequence_numeric(cov_values, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_10(
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
            raise TypeError(f"Covariate names must be strings, got {type(None)} for key")

        if cov_name not in expected_covariates:
            raise ValueError(f"Unknown covariate '{cov_name}', expected one of: {expected_covariates}")

        cov_arr = validate_sequence_numeric(cov_values, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_11(
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

        if cov_name in expected_covariates:
            raise ValueError(f"Unknown covariate '{cov_name}', expected one of: {expected_covariates}")

        cov_arr = validate_sequence_numeric(cov_values, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_12(
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
            raise ValueError(None)

        cov_arr = validate_sequence_numeric(cov_values, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_13(
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

        cov_arr = None

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_14(
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

        cov_arr = validate_sequence_numeric(None, f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_15(
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

        cov_arr = validate_sequence_numeric(cov_values, None)

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_16(
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

        cov_arr = validate_sequence_numeric(f"covariate '{cov_name}'")

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_17(
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

        cov_arr = validate_sequence_numeric(cov_values, )

        if len(cov_arr) != t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_18(
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

        if len(cov_arr) == t_length:
            raise ValueError(
                f"Covariate '{cov_name}' length ({len(cov_arr)}) must match time series length ({t_length})"
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_19(
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
                None
            )

        result[cov_name] = cov_arr

    return result


def x_validate_covariates_dict__mutmut_20(
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

        result[cov_name] = None

    return result

x_validate_covariates_dict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_validate_covariates_dict__mutmut_1': x_validate_covariates_dict__mutmut_1, 
    'x_validate_covariates_dict__mutmut_2': x_validate_covariates_dict__mutmut_2, 
    'x_validate_covariates_dict__mutmut_3': x_validate_covariates_dict__mutmut_3, 
    'x_validate_covariates_dict__mutmut_4': x_validate_covariates_dict__mutmut_4, 
    'x_validate_covariates_dict__mutmut_5': x_validate_covariates_dict__mutmut_5, 
    'x_validate_covariates_dict__mutmut_6': x_validate_covariates_dict__mutmut_6, 
    'x_validate_covariates_dict__mutmut_7': x_validate_covariates_dict__mutmut_7, 
    'x_validate_covariates_dict__mutmut_8': x_validate_covariates_dict__mutmut_8, 
    'x_validate_covariates_dict__mutmut_9': x_validate_covariates_dict__mutmut_9, 
    'x_validate_covariates_dict__mutmut_10': x_validate_covariates_dict__mutmut_10, 
    'x_validate_covariates_dict__mutmut_11': x_validate_covariates_dict__mutmut_11, 
    'x_validate_covariates_dict__mutmut_12': x_validate_covariates_dict__mutmut_12, 
    'x_validate_covariates_dict__mutmut_13': x_validate_covariates_dict__mutmut_13, 
    'x_validate_covariates_dict__mutmut_14': x_validate_covariates_dict__mutmut_14, 
    'x_validate_covariates_dict__mutmut_15': x_validate_covariates_dict__mutmut_15, 
    'x_validate_covariates_dict__mutmut_16': x_validate_covariates_dict__mutmut_16, 
    'x_validate_covariates_dict__mutmut_17': x_validate_covariates_dict__mutmut_17, 
    'x_validate_covariates_dict__mutmut_18': x_validate_covariates_dict__mutmut_18, 
    'x_validate_covariates_dict__mutmut_19': x_validate_covariates_dict__mutmut_19, 
    'x_validate_covariates_dict__mutmut_20': x_validate_covariates_dict__mutmut_20
}
x_validate_covariates_dict__mutmut_orig.__name__ = 'x_validate_covariates_dict'
