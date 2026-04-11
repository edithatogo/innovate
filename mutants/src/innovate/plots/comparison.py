from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
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


def plot_scenario_comparison(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    args = [predictions, title, xlabel, ylabel, cumulative]# type: ignore
    kwargs = {**kwargs}# type: ignore
    return _mutmut_trampoline(x_plot_scenario_comparison__mutmut_orig, x_plot_scenario_comparison__mutmut_mutants, args, kwargs, None)


def x_plot_scenario_comparison__mutmut_orig(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_1(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "XXScenario ComparisonXX",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_2(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "scenario comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_3(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "SCENARIO COMPARISON",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_4(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "XXTimeXX",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_5(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_6(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "TIME",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_7(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "XXCumulative AdoptionsXX",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_8(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "cumulative adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_9(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "CUMULATIVE ADOPTIONS",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_10(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = False,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_11(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=None)

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_12(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(13, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_13(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 8))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_14(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = None
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_15(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    None,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_16(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    None,
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_17(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=None,
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_18(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_19(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_20(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_21(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_22(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = None
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_23(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(None)
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_24(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(None, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_25(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, None, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_26(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=None, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_27(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_28(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_29(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_30(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, )
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_31(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                None,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_32(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "XXPrediction data must be a pandas DataFrame or a sequence of floats.XX",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_33(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "prediction data must be a pandas dataframe or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_34(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "PREDICTION DATA MUST BE A PANDAS DATAFRAME OR A SEQUENCE OF FLOATS.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_35(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_36(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_37(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(None)
    plt.legend()
    plt.grid(True)
    plt.show()


def x_plot_scenario_comparison__mutmut_38(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(None)
    plt.show()


def x_plot_scenario_comparison__mutmut_39(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
    ----
        predictions: A dictionary where keys are scenario names (str) and values
                     are either pandas DataFrames (for multi-product models)
                     or sequences of floats (for single-product models).
                     For DataFrames, the index is assumed to be time.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        cumulative: If True, assumes cumulative adoption. If False, plots rates.
        kwargs: Additional keyword arguments passed to plt.plot.
    """
    plt.figure(figsize=(12, 7))

    for scenario_name, data in predictions.items():
        if isinstance(data, pd.DataFrame):
            # Handle multi-product DataFrame
            time_points = data.index
            for col in data.columns:
                plt.plot(
                    time_points,
                    data[col],
                    label=f"{scenario_name}: {col}",
                    **kwargs,
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats.",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(False)
    plt.show()

x_plot_scenario_comparison__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_scenario_comparison__mutmut_1': x_plot_scenario_comparison__mutmut_1, 
    'x_plot_scenario_comparison__mutmut_2': x_plot_scenario_comparison__mutmut_2, 
    'x_plot_scenario_comparison__mutmut_3': x_plot_scenario_comparison__mutmut_3, 
    'x_plot_scenario_comparison__mutmut_4': x_plot_scenario_comparison__mutmut_4, 
    'x_plot_scenario_comparison__mutmut_5': x_plot_scenario_comparison__mutmut_5, 
    'x_plot_scenario_comparison__mutmut_6': x_plot_scenario_comparison__mutmut_6, 
    'x_plot_scenario_comparison__mutmut_7': x_plot_scenario_comparison__mutmut_7, 
    'x_plot_scenario_comparison__mutmut_8': x_plot_scenario_comparison__mutmut_8, 
    'x_plot_scenario_comparison__mutmut_9': x_plot_scenario_comparison__mutmut_9, 
    'x_plot_scenario_comparison__mutmut_10': x_plot_scenario_comparison__mutmut_10, 
    'x_plot_scenario_comparison__mutmut_11': x_plot_scenario_comparison__mutmut_11, 
    'x_plot_scenario_comparison__mutmut_12': x_plot_scenario_comparison__mutmut_12, 
    'x_plot_scenario_comparison__mutmut_13': x_plot_scenario_comparison__mutmut_13, 
    'x_plot_scenario_comparison__mutmut_14': x_plot_scenario_comparison__mutmut_14, 
    'x_plot_scenario_comparison__mutmut_15': x_plot_scenario_comparison__mutmut_15, 
    'x_plot_scenario_comparison__mutmut_16': x_plot_scenario_comparison__mutmut_16, 
    'x_plot_scenario_comparison__mutmut_17': x_plot_scenario_comparison__mutmut_17, 
    'x_plot_scenario_comparison__mutmut_18': x_plot_scenario_comparison__mutmut_18, 
    'x_plot_scenario_comparison__mutmut_19': x_plot_scenario_comparison__mutmut_19, 
    'x_plot_scenario_comparison__mutmut_20': x_plot_scenario_comparison__mutmut_20, 
    'x_plot_scenario_comparison__mutmut_21': x_plot_scenario_comparison__mutmut_21, 
    'x_plot_scenario_comparison__mutmut_22': x_plot_scenario_comparison__mutmut_22, 
    'x_plot_scenario_comparison__mutmut_23': x_plot_scenario_comparison__mutmut_23, 
    'x_plot_scenario_comparison__mutmut_24': x_plot_scenario_comparison__mutmut_24, 
    'x_plot_scenario_comparison__mutmut_25': x_plot_scenario_comparison__mutmut_25, 
    'x_plot_scenario_comparison__mutmut_26': x_plot_scenario_comparison__mutmut_26, 
    'x_plot_scenario_comparison__mutmut_27': x_plot_scenario_comparison__mutmut_27, 
    'x_plot_scenario_comparison__mutmut_28': x_plot_scenario_comparison__mutmut_28, 
    'x_plot_scenario_comparison__mutmut_29': x_plot_scenario_comparison__mutmut_29, 
    'x_plot_scenario_comparison__mutmut_30': x_plot_scenario_comparison__mutmut_30, 
    'x_plot_scenario_comparison__mutmut_31': x_plot_scenario_comparison__mutmut_31, 
    'x_plot_scenario_comparison__mutmut_32': x_plot_scenario_comparison__mutmut_32, 
    'x_plot_scenario_comparison__mutmut_33': x_plot_scenario_comparison__mutmut_33, 
    'x_plot_scenario_comparison__mutmut_34': x_plot_scenario_comparison__mutmut_34, 
    'x_plot_scenario_comparison__mutmut_35': x_plot_scenario_comparison__mutmut_35, 
    'x_plot_scenario_comparison__mutmut_36': x_plot_scenario_comparison__mutmut_36, 
    'x_plot_scenario_comparison__mutmut_37': x_plot_scenario_comparison__mutmut_37, 
    'x_plot_scenario_comparison__mutmut_38': x_plot_scenario_comparison__mutmut_38, 
    'x_plot_scenario_comparison__mutmut_39': x_plot_scenario_comparison__mutmut_39
}
x_plot_scenario_comparison__mutmut_orig.__name__ = 'x_plot_scenario_comparison'
