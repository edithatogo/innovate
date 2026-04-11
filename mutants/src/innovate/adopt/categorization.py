from collections.abc import Sequence

import pandas as pd

from innovate.backend import current_backend as B
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


def categorize_adopters(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    args = [model, t]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_categorize_adopters__mutmut_orig, x_categorize_adopters__mutmut_mutants, args, kwargs, None)


def x_categorize_adopters__mutmut_orig(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_1(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = None

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_2(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(None)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_3(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = None
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_4(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) * B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_5(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(None) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_6(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t / adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_7(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(None)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_8(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = None

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_9(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        None,
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_10(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) * B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_11(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(None) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_12(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) / adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_13(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) * 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_14(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t + mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_15(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 3) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_16(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(None),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_17(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = None
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_18(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time + 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_19(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 / std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_20(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 3 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_21(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = None
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_22(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time + std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_23(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = None
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_24(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = None

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_25(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time - std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_26(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = None
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_27(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point < innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_28(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append(None)
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_29(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("XXInnovatorsXX")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_30(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_31(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("INNOVATORS")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_32(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point < early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_33(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append(None)
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_34(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("XXEarly AdoptersXX")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_35(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("early adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_36(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("EARLY ADOPTERS")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_37(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point < early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_38(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append(None)
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_39(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("XXEarly MajorityXX")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_40(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("early majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_41(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("EARLY MAJORITY")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_42(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point < late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_43(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append(None)
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_44(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("XXLate MajorityXX")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_45(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("late majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_46(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("LATE MAJORITY")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_47(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append(None)

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_48(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("XXLaggardsXX")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_49(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_50(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("LAGGARDS")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_51(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        None,
    )


def x_categorize_adopters__mutmut_52(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"XXtimeXX": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_53(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"TIME": t, "adoption_rate": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_54(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "XXadoption_rateXX": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_55(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "ADOPTION_RATE": adoption_rate, "category": categories},
    )


def x_categorize_adopters__mutmut_56(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "XXcategoryXX": categories},
    )


def x_categorize_adopters__mutmut_57(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
    ----
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns
    -------
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)

    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(
        B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate),
    )

    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time

    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")

    return pd.DataFrame(
        {"time": t, "adoption_rate": adoption_rate, "CATEGORY": categories},
    )

x_categorize_adopters__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_categorize_adopters__mutmut_1': x_categorize_adopters__mutmut_1, 
    'x_categorize_adopters__mutmut_2': x_categorize_adopters__mutmut_2, 
    'x_categorize_adopters__mutmut_3': x_categorize_adopters__mutmut_3, 
    'x_categorize_adopters__mutmut_4': x_categorize_adopters__mutmut_4, 
    'x_categorize_adopters__mutmut_5': x_categorize_adopters__mutmut_5, 
    'x_categorize_adopters__mutmut_6': x_categorize_adopters__mutmut_6, 
    'x_categorize_adopters__mutmut_7': x_categorize_adopters__mutmut_7, 
    'x_categorize_adopters__mutmut_8': x_categorize_adopters__mutmut_8, 
    'x_categorize_adopters__mutmut_9': x_categorize_adopters__mutmut_9, 
    'x_categorize_adopters__mutmut_10': x_categorize_adopters__mutmut_10, 
    'x_categorize_adopters__mutmut_11': x_categorize_adopters__mutmut_11, 
    'x_categorize_adopters__mutmut_12': x_categorize_adopters__mutmut_12, 
    'x_categorize_adopters__mutmut_13': x_categorize_adopters__mutmut_13, 
    'x_categorize_adopters__mutmut_14': x_categorize_adopters__mutmut_14, 
    'x_categorize_adopters__mutmut_15': x_categorize_adopters__mutmut_15, 
    'x_categorize_adopters__mutmut_16': x_categorize_adopters__mutmut_16, 
    'x_categorize_adopters__mutmut_17': x_categorize_adopters__mutmut_17, 
    'x_categorize_adopters__mutmut_18': x_categorize_adopters__mutmut_18, 
    'x_categorize_adopters__mutmut_19': x_categorize_adopters__mutmut_19, 
    'x_categorize_adopters__mutmut_20': x_categorize_adopters__mutmut_20, 
    'x_categorize_adopters__mutmut_21': x_categorize_adopters__mutmut_21, 
    'x_categorize_adopters__mutmut_22': x_categorize_adopters__mutmut_22, 
    'x_categorize_adopters__mutmut_23': x_categorize_adopters__mutmut_23, 
    'x_categorize_adopters__mutmut_24': x_categorize_adopters__mutmut_24, 
    'x_categorize_adopters__mutmut_25': x_categorize_adopters__mutmut_25, 
    'x_categorize_adopters__mutmut_26': x_categorize_adopters__mutmut_26, 
    'x_categorize_adopters__mutmut_27': x_categorize_adopters__mutmut_27, 
    'x_categorize_adopters__mutmut_28': x_categorize_adopters__mutmut_28, 
    'x_categorize_adopters__mutmut_29': x_categorize_adopters__mutmut_29, 
    'x_categorize_adopters__mutmut_30': x_categorize_adopters__mutmut_30, 
    'x_categorize_adopters__mutmut_31': x_categorize_adopters__mutmut_31, 
    'x_categorize_adopters__mutmut_32': x_categorize_adopters__mutmut_32, 
    'x_categorize_adopters__mutmut_33': x_categorize_adopters__mutmut_33, 
    'x_categorize_adopters__mutmut_34': x_categorize_adopters__mutmut_34, 
    'x_categorize_adopters__mutmut_35': x_categorize_adopters__mutmut_35, 
    'x_categorize_adopters__mutmut_36': x_categorize_adopters__mutmut_36, 
    'x_categorize_adopters__mutmut_37': x_categorize_adopters__mutmut_37, 
    'x_categorize_adopters__mutmut_38': x_categorize_adopters__mutmut_38, 
    'x_categorize_adopters__mutmut_39': x_categorize_adopters__mutmut_39, 
    'x_categorize_adopters__mutmut_40': x_categorize_adopters__mutmut_40, 
    'x_categorize_adopters__mutmut_41': x_categorize_adopters__mutmut_41, 
    'x_categorize_adopters__mutmut_42': x_categorize_adopters__mutmut_42, 
    'x_categorize_adopters__mutmut_43': x_categorize_adopters__mutmut_43, 
    'x_categorize_adopters__mutmut_44': x_categorize_adopters__mutmut_44, 
    'x_categorize_adopters__mutmut_45': x_categorize_adopters__mutmut_45, 
    'x_categorize_adopters__mutmut_46': x_categorize_adopters__mutmut_46, 
    'x_categorize_adopters__mutmut_47': x_categorize_adopters__mutmut_47, 
    'x_categorize_adopters__mutmut_48': x_categorize_adopters__mutmut_48, 
    'x_categorize_adopters__mutmut_49': x_categorize_adopters__mutmut_49, 
    'x_categorize_adopters__mutmut_50': x_categorize_adopters__mutmut_50, 
    'x_categorize_adopters__mutmut_51': x_categorize_adopters__mutmut_51, 
    'x_categorize_adopters__mutmut_52': x_categorize_adopters__mutmut_52, 
    'x_categorize_adopters__mutmut_53': x_categorize_adopters__mutmut_53, 
    'x_categorize_adopters__mutmut_54': x_categorize_adopters__mutmut_54, 
    'x_categorize_adopters__mutmut_55': x_categorize_adopters__mutmut_55, 
    'x_categorize_adopters__mutmut_56': x_categorize_adopters__mutmut_56, 
    'x_categorize_adopters__mutmut_57': x_categorize_adopters__mutmut_57
}
x_categorize_adopters__mutmut_orig.__name__ = 'x_categorize_adopters'
