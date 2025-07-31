import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Sequence, Union


def plot_scenario_comparison(
    predictions: Dict[str, Union[pd.DataFrame, Sequence[float]]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """
    Plots multiple diffusion scenarios on a single graph for comparison.

    Args:
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
                    time_points, data[col], label=f"{scenario_name}: {col}", **kwargs
                )
        elif isinstance(data, Sequence):
            # Handle single-product sequence (assumes time is 0 to len(data)-1 or provided separately)
            # For simplicity, assume time is implicit 0 to len-1 if not a DataFrame
            time_points = range(len(data))
            plt.plot(time_points, data, label=scenario_name, **kwargs)
        else:
            raise TypeError(
                "Prediction data must be a pandas DataFrame or a sequence of floats."
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
