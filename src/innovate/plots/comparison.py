from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_scenario_comparison(
    predictions: dict[str, pd.DataFrame | Sequence[float]],
    title: str = "Scenario Comparison",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    cumulative: bool = True,
    **kwargs,
):
    """Plot multiple diffusion scenarios on a single graph.

    Parameters
    ----------
    predictions : Mapping[str, pandas.DataFrame | Sequence[float]]
        Mapping from scenario names to either pandas DataFrames for
        multi-product models or sequences of floats for single-product
        models. DataFrame indices are treated as time.
    title : str, default="Scenario Comparison"
        Plot title.
    xlabel : str, default="Time"
        X-axis label.
    ylabel : str, default="Cumulative Adoptions"
        Y-axis label.
    cumulative : bool, default=True
        If True, plot cumulative adoption; otherwise plot rates.
    **kwargs
        Additional keyword arguments forwarded to ``plt.plot``.
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
