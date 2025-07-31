import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence, Optional


def plot_diffusion_curve(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: Optional[str] = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_multi_product_diffusion(
    df_pred: pd.DataFrame,
    df_obs: Optional[pd.DataFrame] = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: Optional[str] = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
        df_pred: DataFrame of predicted cumulative adoptions (index is time, columns are product names).
        df_obs: Optional DataFrame of observed cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(12, 7))

    # Plot predicted curves
    for col in df_pred.columns:
        plt.plot(
            df_pred.index, df_pred[col], "-", label=f"Predicted {col}", linewidth=2
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
