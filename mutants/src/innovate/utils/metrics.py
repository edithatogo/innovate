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


def calculate_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_mse__mutmut_orig, x_calculate_mse__mutmut_mutants, args, kwargs, None)


def x_calculate_mse__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = None
    return float(result)


def x_calculate_mse__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(None)
    return float(result)


def x_calculate_mse__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) * 2)
    return float(result)


def x_calculate_mse__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr + y_pred_arr) ** 2)
    return float(result)


def x_calculate_mse__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 3)
    return float(result)


def x_calculate_mse__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean((y_true_arr - y_pred_arr) ** 2)
    return float(None)

x_calculate_mse__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_mse__mutmut_1': x_calculate_mse__mutmut_1, 
    'x_calculate_mse__mutmut_2': x_calculate_mse__mutmut_2, 
    'x_calculate_mse__mutmut_3': x_calculate_mse__mutmut_3, 
    'x_calculate_mse__mutmut_4': x_calculate_mse__mutmut_4, 
    'x_calculate_mse__mutmut_5': x_calculate_mse__mutmut_5, 
    'x_calculate_mse__mutmut_6': x_calculate_mse__mutmut_6, 
    'x_calculate_mse__mutmut_7': x_calculate_mse__mutmut_7, 
    'x_calculate_mse__mutmut_8': x_calculate_mse__mutmut_8, 
    'x_calculate_mse__mutmut_9': x_calculate_mse__mutmut_9, 
    'x_calculate_mse__mutmut_10': x_calculate_mse__mutmut_10, 
    'x_calculate_mse__mutmut_11': x_calculate_mse__mutmut_11, 
    'x_calculate_mse__mutmut_12': x_calculate_mse__mutmut_12, 
    'x_calculate_mse__mutmut_13': x_calculate_mse__mutmut_13, 
    'x_calculate_mse__mutmut_14': x_calculate_mse__mutmut_14, 
    'x_calculate_mse__mutmut_15': x_calculate_mse__mutmut_15, 
    'x_calculate_mse__mutmut_16': x_calculate_mse__mutmut_16
}
x_calculate_mse__mutmut_orig.__name__ = 'x_calculate_mse'


def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_rmse__mutmut_orig, x_calculate_rmse__mutmut_mutants, args, kwargs, None)


def x_calculate_rmse__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = None
    return float(result)


def x_calculate_rmse__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(None)
    return float(result)


def x_calculate_rmse__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean(None))
    return float(result)


def x_calculate_rmse__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) * 2))
    return float(result)


def x_calculate_rmse__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr + y_pred_arr) ** 2))
    return float(result)


def x_calculate_rmse__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 3))
    return float(result)


def x_calculate_rmse__mutmut_17(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    return float(None)

x_calculate_rmse__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_rmse__mutmut_1': x_calculate_rmse__mutmut_1, 
    'x_calculate_rmse__mutmut_2': x_calculate_rmse__mutmut_2, 
    'x_calculate_rmse__mutmut_3': x_calculate_rmse__mutmut_3, 
    'x_calculate_rmse__mutmut_4': x_calculate_rmse__mutmut_4, 
    'x_calculate_rmse__mutmut_5': x_calculate_rmse__mutmut_5, 
    'x_calculate_rmse__mutmut_6': x_calculate_rmse__mutmut_6, 
    'x_calculate_rmse__mutmut_7': x_calculate_rmse__mutmut_7, 
    'x_calculate_rmse__mutmut_8': x_calculate_rmse__mutmut_8, 
    'x_calculate_rmse__mutmut_9': x_calculate_rmse__mutmut_9, 
    'x_calculate_rmse__mutmut_10': x_calculate_rmse__mutmut_10, 
    'x_calculate_rmse__mutmut_11': x_calculate_rmse__mutmut_11, 
    'x_calculate_rmse__mutmut_12': x_calculate_rmse__mutmut_12, 
    'x_calculate_rmse__mutmut_13': x_calculate_rmse__mutmut_13, 
    'x_calculate_rmse__mutmut_14': x_calculate_rmse__mutmut_14, 
    'x_calculate_rmse__mutmut_15': x_calculate_rmse__mutmut_15, 
    'x_calculate_rmse__mutmut_16': x_calculate_rmse__mutmut_16, 
    'x_calculate_rmse__mutmut_17': x_calculate_rmse__mutmut_17
}
x_calculate_rmse__mutmut_orig.__name__ = 'x_calculate_rmse'


def calculate_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_mape__mutmut_orig, x_calculate_mape__mutmut_mutants, args, kwargs, None)


def x_calculate_mape__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = None
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr == 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 1
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(None):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(None)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_17(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = None
    return float(result)


def x_calculate_mape__mutmut_18(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        ) / 100
    )
    return float(result)


def x_calculate_mape__mutmut_19(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            None,
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_20(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                None,
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_21(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) * y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_22(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] + y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(result)


def x_calculate_mape__mutmut_23(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 101
    )
    return float(result)


def x_calculate_mape__mutmut_24(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # Avoid division by zero for true values that are zero
    non_zero_mask = y_true_arr != 0
    if not np.any(non_zero_mask):
        return float(np.nan)  # Or raise an error, depending on desired behavior
    result = (
        np.mean(
            np.abs(
                (y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask],
            ),
        )
        * 100
    )
    return float(None)

x_calculate_mape__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_mape__mutmut_1': x_calculate_mape__mutmut_1, 
    'x_calculate_mape__mutmut_2': x_calculate_mape__mutmut_2, 
    'x_calculate_mape__mutmut_3': x_calculate_mape__mutmut_3, 
    'x_calculate_mape__mutmut_4': x_calculate_mape__mutmut_4, 
    'x_calculate_mape__mutmut_5': x_calculate_mape__mutmut_5, 
    'x_calculate_mape__mutmut_6': x_calculate_mape__mutmut_6, 
    'x_calculate_mape__mutmut_7': x_calculate_mape__mutmut_7, 
    'x_calculate_mape__mutmut_8': x_calculate_mape__mutmut_8, 
    'x_calculate_mape__mutmut_9': x_calculate_mape__mutmut_9, 
    'x_calculate_mape__mutmut_10': x_calculate_mape__mutmut_10, 
    'x_calculate_mape__mutmut_11': x_calculate_mape__mutmut_11, 
    'x_calculate_mape__mutmut_12': x_calculate_mape__mutmut_12, 
    'x_calculate_mape__mutmut_13': x_calculate_mape__mutmut_13, 
    'x_calculate_mape__mutmut_14': x_calculate_mape__mutmut_14, 
    'x_calculate_mape__mutmut_15': x_calculate_mape__mutmut_15, 
    'x_calculate_mape__mutmut_16': x_calculate_mape__mutmut_16, 
    'x_calculate_mape__mutmut_17': x_calculate_mape__mutmut_17, 
    'x_calculate_mape__mutmut_18': x_calculate_mape__mutmut_18, 
    'x_calculate_mape__mutmut_19': x_calculate_mape__mutmut_19, 
    'x_calculate_mape__mutmut_20': x_calculate_mape__mutmut_20, 
    'x_calculate_mape__mutmut_21': x_calculate_mape__mutmut_21, 
    'x_calculate_mape__mutmut_22': x_calculate_mape__mutmut_22, 
    'x_calculate_mape__mutmut_23': x_calculate_mape__mutmut_23, 
    'x_calculate_mape__mutmut_24': x_calculate_mape__mutmut_24
}
x_calculate_mape__mutmut_orig.__name__ = 'x_calculate_mape'


def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_mae__mutmut_orig, x_calculate_mae__mutmut_mutants, args, kwargs, None)


def x_calculate_mae__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = None
    return float(result)


def x_calculate_mae__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(None)
    return float(result)


def x_calculate_mae__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(None))
    return float(result)


def x_calculate_mae__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr + y_pred_arr))
    return float(result)


def x_calculate_mae__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.mean(np.abs(y_true_arr - y_pred_arr))
    return float(None)

x_calculate_mae__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_mae__mutmut_1': x_calculate_mae__mutmut_1, 
    'x_calculate_mae__mutmut_2': x_calculate_mae__mutmut_2, 
    'x_calculate_mae__mutmut_3': x_calculate_mae__mutmut_3, 
    'x_calculate_mae__mutmut_4': x_calculate_mae__mutmut_4, 
    'x_calculate_mae__mutmut_5': x_calculate_mae__mutmut_5, 
    'x_calculate_mae__mutmut_6': x_calculate_mae__mutmut_6, 
    'x_calculate_mae__mutmut_7': x_calculate_mae__mutmut_7, 
    'x_calculate_mae__mutmut_8': x_calculate_mae__mutmut_8, 
    'x_calculate_mae__mutmut_9': x_calculate_mae__mutmut_9, 
    'x_calculate_mae__mutmut_10': x_calculate_mae__mutmut_10, 
    'x_calculate_mae__mutmut_11': x_calculate_mae__mutmut_11, 
    'x_calculate_mae__mutmut_12': x_calculate_mae__mutmut_12, 
    'x_calculate_mae__mutmut_13': x_calculate_mae__mutmut_13, 
    'x_calculate_mae__mutmut_14': x_calculate_mae__mutmut_14, 
    'x_calculate_mae__mutmut_15': x_calculate_mae__mutmut_15
}
x_calculate_mae__mutmut_orig.__name__ = 'x_calculate_mae'


def calculate_r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_r_squared__mutmut_orig, x_calculate_r_squared__mutmut_mutants, args, kwargs, None)


def x_calculate_r_squared__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = None
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum(None)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) * 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr + y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 3)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = None
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_17(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum(None)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_18(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) * 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_19(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr + np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_20(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(None)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_21(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 3)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_22(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = None
    return float(result)


def x_calculate_r_squared__mutmut_23(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_24(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_25(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_26(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_27(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0
    return float(result)


def x_calculate_r_squared__mutmut_28(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(result)


def x_calculate_r_squared__mutmut_29(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the R-squared (coefficient of determination)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    result = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(None)

x_calculate_r_squared__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_r_squared__mutmut_1': x_calculate_r_squared__mutmut_1, 
    'x_calculate_r_squared__mutmut_2': x_calculate_r_squared__mutmut_2, 
    'x_calculate_r_squared__mutmut_3': x_calculate_r_squared__mutmut_3, 
    'x_calculate_r_squared__mutmut_4': x_calculate_r_squared__mutmut_4, 
    'x_calculate_r_squared__mutmut_5': x_calculate_r_squared__mutmut_5, 
    'x_calculate_r_squared__mutmut_6': x_calculate_r_squared__mutmut_6, 
    'x_calculate_r_squared__mutmut_7': x_calculate_r_squared__mutmut_7, 
    'x_calculate_r_squared__mutmut_8': x_calculate_r_squared__mutmut_8, 
    'x_calculate_r_squared__mutmut_9': x_calculate_r_squared__mutmut_9, 
    'x_calculate_r_squared__mutmut_10': x_calculate_r_squared__mutmut_10, 
    'x_calculate_r_squared__mutmut_11': x_calculate_r_squared__mutmut_11, 
    'x_calculate_r_squared__mutmut_12': x_calculate_r_squared__mutmut_12, 
    'x_calculate_r_squared__mutmut_13': x_calculate_r_squared__mutmut_13, 
    'x_calculate_r_squared__mutmut_14': x_calculate_r_squared__mutmut_14, 
    'x_calculate_r_squared__mutmut_15': x_calculate_r_squared__mutmut_15, 
    'x_calculate_r_squared__mutmut_16': x_calculate_r_squared__mutmut_16, 
    'x_calculate_r_squared__mutmut_17': x_calculate_r_squared__mutmut_17, 
    'x_calculate_r_squared__mutmut_18': x_calculate_r_squared__mutmut_18, 
    'x_calculate_r_squared__mutmut_19': x_calculate_r_squared__mutmut_19, 
    'x_calculate_r_squared__mutmut_20': x_calculate_r_squared__mutmut_20, 
    'x_calculate_r_squared__mutmut_21': x_calculate_r_squared__mutmut_21, 
    'x_calculate_r_squared__mutmut_22': x_calculate_r_squared__mutmut_22, 
    'x_calculate_r_squared__mutmut_23': x_calculate_r_squared__mutmut_23, 
    'x_calculate_r_squared__mutmut_24': x_calculate_r_squared__mutmut_24, 
    'x_calculate_r_squared__mutmut_25': x_calculate_r_squared__mutmut_25, 
    'x_calculate_r_squared__mutmut_26': x_calculate_r_squared__mutmut_26, 
    'x_calculate_r_squared__mutmut_27': x_calculate_r_squared__mutmut_27, 
    'x_calculate_r_squared__mutmut_28': x_calculate_r_squared__mutmut_28, 
    'x_calculate_r_squared__mutmut_29': x_calculate_r_squared__mutmut_29
}
x_calculate_r_squared__mutmut_orig.__name__ = 'x_calculate_r_squared'


def calculate_smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_smape__mutmut_orig, x_calculate_smape__mutmut_mutants, args, kwargs, None)


def x_calculate_smape__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = None
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(None)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr + y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = None
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) * 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) - np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_17(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(None) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_18(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(None)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_19(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 3
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_20(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = None
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_21(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator == 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_22(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 1
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_23(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_24(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(None):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_25(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 1.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_26(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(None) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_27(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr != y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_28(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(None)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_29(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = None
    return float(result)


def x_calculate_smape__mutmut_30(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) / 100
    return float(result)


def x_calculate_smape__mutmut_31(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(None) * 100
    return float(result)


def x_calculate_smape__mutmut_32(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] * denominator[non_zero_mask]) * 100
    return float(result)


def x_calculate_smape__mutmut_33(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 101
    return float(result)


def x_calculate_smape__mutmut_34(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    numerator = np.abs(y_pred_arr - y_true_arr)
    denominator = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    # Avoid division by zero
    non_zero_mask = denominator != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_true_arr == y_pred_arr) else float(np.nan)
    result = np.mean(numerator[non_zero_mask] / denominator[non_zero_mask]) * 100
    return float(None)

x_calculate_smape__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_smape__mutmut_1': x_calculate_smape__mutmut_1, 
    'x_calculate_smape__mutmut_2': x_calculate_smape__mutmut_2, 
    'x_calculate_smape__mutmut_3': x_calculate_smape__mutmut_3, 
    'x_calculate_smape__mutmut_4': x_calculate_smape__mutmut_4, 
    'x_calculate_smape__mutmut_5': x_calculate_smape__mutmut_5, 
    'x_calculate_smape__mutmut_6': x_calculate_smape__mutmut_6, 
    'x_calculate_smape__mutmut_7': x_calculate_smape__mutmut_7, 
    'x_calculate_smape__mutmut_8': x_calculate_smape__mutmut_8, 
    'x_calculate_smape__mutmut_9': x_calculate_smape__mutmut_9, 
    'x_calculate_smape__mutmut_10': x_calculate_smape__mutmut_10, 
    'x_calculate_smape__mutmut_11': x_calculate_smape__mutmut_11, 
    'x_calculate_smape__mutmut_12': x_calculate_smape__mutmut_12, 
    'x_calculate_smape__mutmut_13': x_calculate_smape__mutmut_13, 
    'x_calculate_smape__mutmut_14': x_calculate_smape__mutmut_14, 
    'x_calculate_smape__mutmut_15': x_calculate_smape__mutmut_15, 
    'x_calculate_smape__mutmut_16': x_calculate_smape__mutmut_16, 
    'x_calculate_smape__mutmut_17': x_calculate_smape__mutmut_17, 
    'x_calculate_smape__mutmut_18': x_calculate_smape__mutmut_18, 
    'x_calculate_smape__mutmut_19': x_calculate_smape__mutmut_19, 
    'x_calculate_smape__mutmut_20': x_calculate_smape__mutmut_20, 
    'x_calculate_smape__mutmut_21': x_calculate_smape__mutmut_21, 
    'x_calculate_smape__mutmut_22': x_calculate_smape__mutmut_22, 
    'x_calculate_smape__mutmut_23': x_calculate_smape__mutmut_23, 
    'x_calculate_smape__mutmut_24': x_calculate_smape__mutmut_24, 
    'x_calculate_smape__mutmut_25': x_calculate_smape__mutmut_25, 
    'x_calculate_smape__mutmut_26': x_calculate_smape__mutmut_26, 
    'x_calculate_smape__mutmut_27': x_calculate_smape__mutmut_27, 
    'x_calculate_smape__mutmut_28': x_calculate_smape__mutmut_28, 
    'x_calculate_smape__mutmut_29': x_calculate_smape__mutmut_29, 
    'x_calculate_smape__mutmut_30': x_calculate_smape__mutmut_30, 
    'x_calculate_smape__mutmut_31': x_calculate_smape__mutmut_31, 
    'x_calculate_smape__mutmut_32': x_calculate_smape__mutmut_32, 
    'x_calculate_smape__mutmut_33': x_calculate_smape__mutmut_33, 
    'x_calculate_smape__mutmut_34': x_calculate_smape__mutmut_34
}
x_calculate_smape__mutmut_orig.__name__ = 'x_calculate_smape'


def calculate_rss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    args = [y_true, y_pred]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_rss__mutmut_orig, x_calculate_rss__mutmut_mutants, args, kwargs, None)


def x_calculate_rss__mutmut_orig(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_1(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = None
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(None, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_3(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=None)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_4(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_5(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, )
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_6(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = None
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_7(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(None, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_8(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=None)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_9(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_10(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, )
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_11(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = None
    return float(result)


def x_calculate_rss__mutmut_12(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum(None)
    return float(result)


def x_calculate_rss__mutmut_13(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) * 2)
    return float(result)


def x_calculate_rss__mutmut_14(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr + y_pred_arr) ** 2)
    return float(result)


def x_calculate_rss__mutmut_15(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 3)
    return float(result)


def x_calculate_rss__mutmut_16(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    result = np.sum((y_true_arr - y_pred_arr) ** 2)
    return float(None)

x_calculate_rss__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_rss__mutmut_1': x_calculate_rss__mutmut_1, 
    'x_calculate_rss__mutmut_2': x_calculate_rss__mutmut_2, 
    'x_calculate_rss__mutmut_3': x_calculate_rss__mutmut_3, 
    'x_calculate_rss__mutmut_4': x_calculate_rss__mutmut_4, 
    'x_calculate_rss__mutmut_5': x_calculate_rss__mutmut_5, 
    'x_calculate_rss__mutmut_6': x_calculate_rss__mutmut_6, 
    'x_calculate_rss__mutmut_7': x_calculate_rss__mutmut_7, 
    'x_calculate_rss__mutmut_8': x_calculate_rss__mutmut_8, 
    'x_calculate_rss__mutmut_9': x_calculate_rss__mutmut_9, 
    'x_calculate_rss__mutmut_10': x_calculate_rss__mutmut_10, 
    'x_calculate_rss__mutmut_11': x_calculate_rss__mutmut_11, 
    'x_calculate_rss__mutmut_12': x_calculate_rss__mutmut_12, 
    'x_calculate_rss__mutmut_13': x_calculate_rss__mutmut_13, 
    'x_calculate_rss__mutmut_14': x_calculate_rss__mutmut_14, 
    'x_calculate_rss__mutmut_15': x_calculate_rss__mutmut_15, 
    'x_calculate_rss__mutmut_16': x_calculate_rss__mutmut_16
}
x_calculate_rss__mutmut_orig.__name__ = 'x_calculate_rss'


def calculate_aic(n_params: int, n_samples: int, rss: float) -> float:
    args = [n_params, n_samples, rss]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_aic__mutmut_orig, x_calculate_aic__mutmut_mutants, args, kwargs, None)


def x_calculate_aic__mutmut_orig(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_1(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 and rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_2(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples != 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_3(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 1 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_4(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss < 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_5(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 1:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_6(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(None)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_7(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = None
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_8(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) + n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_9(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) + n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_10(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 / np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_11(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples * 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_12(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = +n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_13(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 3 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_14(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(None) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_15(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 / np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_16(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(3 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_17(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 / np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_18(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples * 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_19(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 3 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_20(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(None) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_21(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss * n_samples) - n_samples / 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_22(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples * 2
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_23(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 3
    return float(2 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_24(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(None)


def x_calculate_aic__mutmut_25(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params + 2 * log_likelihood)


def x_calculate_aic__mutmut_26(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 / n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_27(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(3 * n_params - 2 * log_likelihood)


def x_calculate_aic__mutmut_28(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 2 / log_likelihood)


def x_calculate_aic__mutmut_29(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Akaike Information Criterion (AIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(2 * n_params - 3 * log_likelihood)

x_calculate_aic__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_aic__mutmut_1': x_calculate_aic__mutmut_1, 
    'x_calculate_aic__mutmut_2': x_calculate_aic__mutmut_2, 
    'x_calculate_aic__mutmut_3': x_calculate_aic__mutmut_3, 
    'x_calculate_aic__mutmut_4': x_calculate_aic__mutmut_4, 
    'x_calculate_aic__mutmut_5': x_calculate_aic__mutmut_5, 
    'x_calculate_aic__mutmut_6': x_calculate_aic__mutmut_6, 
    'x_calculate_aic__mutmut_7': x_calculate_aic__mutmut_7, 
    'x_calculate_aic__mutmut_8': x_calculate_aic__mutmut_8, 
    'x_calculate_aic__mutmut_9': x_calculate_aic__mutmut_9, 
    'x_calculate_aic__mutmut_10': x_calculate_aic__mutmut_10, 
    'x_calculate_aic__mutmut_11': x_calculate_aic__mutmut_11, 
    'x_calculate_aic__mutmut_12': x_calculate_aic__mutmut_12, 
    'x_calculate_aic__mutmut_13': x_calculate_aic__mutmut_13, 
    'x_calculate_aic__mutmut_14': x_calculate_aic__mutmut_14, 
    'x_calculate_aic__mutmut_15': x_calculate_aic__mutmut_15, 
    'x_calculate_aic__mutmut_16': x_calculate_aic__mutmut_16, 
    'x_calculate_aic__mutmut_17': x_calculate_aic__mutmut_17, 
    'x_calculate_aic__mutmut_18': x_calculate_aic__mutmut_18, 
    'x_calculate_aic__mutmut_19': x_calculate_aic__mutmut_19, 
    'x_calculate_aic__mutmut_20': x_calculate_aic__mutmut_20, 
    'x_calculate_aic__mutmut_21': x_calculate_aic__mutmut_21, 
    'x_calculate_aic__mutmut_22': x_calculate_aic__mutmut_22, 
    'x_calculate_aic__mutmut_23': x_calculate_aic__mutmut_23, 
    'x_calculate_aic__mutmut_24': x_calculate_aic__mutmut_24, 
    'x_calculate_aic__mutmut_25': x_calculate_aic__mutmut_25, 
    'x_calculate_aic__mutmut_26': x_calculate_aic__mutmut_26, 
    'x_calculate_aic__mutmut_27': x_calculate_aic__mutmut_27, 
    'x_calculate_aic__mutmut_28': x_calculate_aic__mutmut_28, 
    'x_calculate_aic__mutmut_29': x_calculate_aic__mutmut_29
}
x_calculate_aic__mutmut_orig.__name__ = 'x_calculate_aic'


def calculate_bic(n_params: int, n_samples: int, rss: float) -> float:
    args = [n_params, n_samples, rss]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_calculate_bic__mutmut_orig, x_calculate_bic__mutmut_mutants, args, kwargs, None)


def x_calculate_bic__mutmut_orig(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_1(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 and rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_2(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples != 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_3(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 1 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_4(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss < 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_5(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 1:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_6(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(None)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_7(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = None
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_8(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) + n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_9(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) + n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_10(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 / np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_11(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples * 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_12(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = +n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_13(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 3 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_14(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(None) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_15(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 / np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_16(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(3 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_17(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 / np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_18(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples * 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_19(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 3 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_20(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(None) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_21(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss * n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_22(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples * 2
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_23(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 3
    return float(n_params * np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_24(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(None)


def x_calculate_bic__mutmut_25(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) + 2 * log_likelihood)


def x_calculate_bic__mutmut_26(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params / np.log(n_samples) - 2 * log_likelihood)


def x_calculate_bic__mutmut_27(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(None) - 2 * log_likelihood)


def x_calculate_bic__mutmut_28(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 2 / log_likelihood)


def x_calculate_bic__mutmut_29(n_params: int, n_samples: int, rss: float) -> float:
    """Calculates the Bayesian Information Criterion (BIC).

    Assumes errors are normally distributed.
    """
    if n_samples == 0 or rss <= 0:
        return float(np.nan)
    log_likelihood = -n_samples / 2 * np.log(2 * np.pi) - n_samples / 2 * np.log(rss / n_samples) - n_samples / 2
    return float(n_params * np.log(n_samples) - 3 * log_likelihood)

x_calculate_bic__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_calculate_bic__mutmut_1': x_calculate_bic__mutmut_1, 
    'x_calculate_bic__mutmut_2': x_calculate_bic__mutmut_2, 
    'x_calculate_bic__mutmut_3': x_calculate_bic__mutmut_3, 
    'x_calculate_bic__mutmut_4': x_calculate_bic__mutmut_4, 
    'x_calculate_bic__mutmut_5': x_calculate_bic__mutmut_5, 
    'x_calculate_bic__mutmut_6': x_calculate_bic__mutmut_6, 
    'x_calculate_bic__mutmut_7': x_calculate_bic__mutmut_7, 
    'x_calculate_bic__mutmut_8': x_calculate_bic__mutmut_8, 
    'x_calculate_bic__mutmut_9': x_calculate_bic__mutmut_9, 
    'x_calculate_bic__mutmut_10': x_calculate_bic__mutmut_10, 
    'x_calculate_bic__mutmut_11': x_calculate_bic__mutmut_11, 
    'x_calculate_bic__mutmut_12': x_calculate_bic__mutmut_12, 
    'x_calculate_bic__mutmut_13': x_calculate_bic__mutmut_13, 
    'x_calculate_bic__mutmut_14': x_calculate_bic__mutmut_14, 
    'x_calculate_bic__mutmut_15': x_calculate_bic__mutmut_15, 
    'x_calculate_bic__mutmut_16': x_calculate_bic__mutmut_16, 
    'x_calculate_bic__mutmut_17': x_calculate_bic__mutmut_17, 
    'x_calculate_bic__mutmut_18': x_calculate_bic__mutmut_18, 
    'x_calculate_bic__mutmut_19': x_calculate_bic__mutmut_19, 
    'x_calculate_bic__mutmut_20': x_calculate_bic__mutmut_20, 
    'x_calculate_bic__mutmut_21': x_calculate_bic__mutmut_21, 
    'x_calculate_bic__mutmut_22': x_calculate_bic__mutmut_22, 
    'x_calculate_bic__mutmut_23': x_calculate_bic__mutmut_23, 
    'x_calculate_bic__mutmut_24': x_calculate_bic__mutmut_24, 
    'x_calculate_bic__mutmut_25': x_calculate_bic__mutmut_25, 
    'x_calculate_bic__mutmut_26': x_calculate_bic__mutmut_26, 
    'x_calculate_bic__mutmut_27': x_calculate_bic__mutmut_27, 
    'x_calculate_bic__mutmut_28': x_calculate_bic__mutmut_28, 
    'x_calculate_bic__mutmut_29': x_calculate_bic__mutmut_29
}
x_calculate_bic__mutmut_orig.__name__ = 'x_calculate_bic'
