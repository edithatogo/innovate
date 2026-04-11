# src/innovate/fail/analysis.py


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


def analyze_failure(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    args = [predictions, failure_threshold, time_horizon]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_analyze_failure__mutmut_orig, x_analyze_failure__mutmut_mutants, args, kwargs, None)


def x_analyze_failure__mutmut_orig(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_1(
    predictions: np.ndarray,
    failure_threshold: float = 1.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_2(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim == 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_3(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 3:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_4(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError(None)

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_5(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("XX`predictions` must be a 2D array.XX")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_6(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2d array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_7(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`PREDICTIONS` MUST BE A 2D ARRAY.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_8(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_9(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (1 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_10(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 <= failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_11(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold <= 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_12(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 2):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_13(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError(None)

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_14(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("XX`failure_threshold` must be between 0 and 1.XX")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_15(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`FAILURE_THRESHOLD` MUST BE BETWEEN 0 AND 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_16(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon != -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_17(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == +1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_18(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -2:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_19(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = None

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_20(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[1]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_21(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_22(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (1 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_23(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 <= time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_24(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon < predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_25(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[1]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_26(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError(None)

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_27(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("XXInvalid `time_horizon`.XX")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_28(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_29(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("INVALID `TIME_HORIZON`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_30(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = None
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_31(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(None):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_32(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[2]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_33(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(None) < failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_34(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) <= failure_threshold:
            failed_indices.append(i)

    return failed_indices


def x_analyze_failure__mutmut_35(
    predictions: np.ndarray,
    failure_threshold: float = 0.1,
    time_horizon: int = -1,
) -> list[int]:
    """Analyzes the results of a competition model to identify failed technologies.

    A technology is considered to have failed if its market share does not
    exceed the failure_threshold within the given time_horizon.

    Args:
    ----
        predictions: A 2D array of market share predictions from a
                     CompetitionModel.
        failure_threshold: The market share threshold for a technology to be
                           considered successful.
        time_horizon: The number of time steps over which to evaluate the
                      failure condition. If -1, the entire time series is
                      considered.

    Returns
    -------
        A list of indices of the technologies that have failed.
    """
    if predictions.ndim != 2:
        raise ValueError("`predictions` must be a 2D array.")

    if not (0 < failure_threshold < 1):
        raise ValueError("`failure_threshold` must be between 0 and 1.")

    if time_horizon == -1:
        time_horizon = predictions.shape[0]

    if not (0 < time_horizon <= predictions.shape[0]):
        raise ValueError("Invalid `time_horizon`.")

    failed_indices = []
    for i in range(predictions.shape[1]):
        if np.max(predictions[:time_horizon, i]) < failure_threshold:
            failed_indices.append(None)

    return failed_indices

x_analyze_failure__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_analyze_failure__mutmut_1': x_analyze_failure__mutmut_1, 
    'x_analyze_failure__mutmut_2': x_analyze_failure__mutmut_2, 
    'x_analyze_failure__mutmut_3': x_analyze_failure__mutmut_3, 
    'x_analyze_failure__mutmut_4': x_analyze_failure__mutmut_4, 
    'x_analyze_failure__mutmut_5': x_analyze_failure__mutmut_5, 
    'x_analyze_failure__mutmut_6': x_analyze_failure__mutmut_6, 
    'x_analyze_failure__mutmut_7': x_analyze_failure__mutmut_7, 
    'x_analyze_failure__mutmut_8': x_analyze_failure__mutmut_8, 
    'x_analyze_failure__mutmut_9': x_analyze_failure__mutmut_9, 
    'x_analyze_failure__mutmut_10': x_analyze_failure__mutmut_10, 
    'x_analyze_failure__mutmut_11': x_analyze_failure__mutmut_11, 
    'x_analyze_failure__mutmut_12': x_analyze_failure__mutmut_12, 
    'x_analyze_failure__mutmut_13': x_analyze_failure__mutmut_13, 
    'x_analyze_failure__mutmut_14': x_analyze_failure__mutmut_14, 
    'x_analyze_failure__mutmut_15': x_analyze_failure__mutmut_15, 
    'x_analyze_failure__mutmut_16': x_analyze_failure__mutmut_16, 
    'x_analyze_failure__mutmut_17': x_analyze_failure__mutmut_17, 
    'x_analyze_failure__mutmut_18': x_analyze_failure__mutmut_18, 
    'x_analyze_failure__mutmut_19': x_analyze_failure__mutmut_19, 
    'x_analyze_failure__mutmut_20': x_analyze_failure__mutmut_20, 
    'x_analyze_failure__mutmut_21': x_analyze_failure__mutmut_21, 
    'x_analyze_failure__mutmut_22': x_analyze_failure__mutmut_22, 
    'x_analyze_failure__mutmut_23': x_analyze_failure__mutmut_23, 
    'x_analyze_failure__mutmut_24': x_analyze_failure__mutmut_24, 
    'x_analyze_failure__mutmut_25': x_analyze_failure__mutmut_25, 
    'x_analyze_failure__mutmut_26': x_analyze_failure__mutmut_26, 
    'x_analyze_failure__mutmut_27': x_analyze_failure__mutmut_27, 
    'x_analyze_failure__mutmut_28': x_analyze_failure__mutmut_28, 
    'x_analyze_failure__mutmut_29': x_analyze_failure__mutmut_29, 
    'x_analyze_failure__mutmut_30': x_analyze_failure__mutmut_30, 
    'x_analyze_failure__mutmut_31': x_analyze_failure__mutmut_31, 
    'x_analyze_failure__mutmut_32': x_analyze_failure__mutmut_32, 
    'x_analyze_failure__mutmut_33': x_analyze_failure__mutmut_33, 
    'x_analyze_failure__mutmut_34': x_analyze_failure__mutmut_34, 
    'x_analyze_failure__mutmut_35': x_analyze_failure__mutmut_35
}
x_analyze_failure__mutmut_orig.__name__ = 'x_analyze_failure'
