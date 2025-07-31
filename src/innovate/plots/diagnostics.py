# src/innovate/plots/diagnostics.py

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


def plot_residuals(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """
    Plots the residuals of a fitted model, along with their ACF and PACF plots.

    Parameters
    ----------
    model : DiffusionModel
        A fitted diffusion model.
    t : np.ndarray
        The time steps.
    y : np.ndarray
        The observed data.
    title : str, optional
        The title for the overall plot, by default "Residual Analysis".
    lags : int, optional
        The number of lags to show in the ACF and PACF plots, by default 30.
    acf_only : bool, optional
        If True, only the ACF plot will be shown, by default False.
    figsize : tuple, optional
        The figure size, by default (10, None). If None, the height is automatically determined.
    color_residuals : str, optional
        The color of the residuals plot, by default 'C0'.
    color_acf : str, optional
        The color of the ACF plot, by default 'C1'.
    color_pacf : str, optional
        The color of the PACF plot, by default 'C2'.
    show : bool, optional
        If True, the plot will be shown, by default True. Otherwise, the figure and axes objects will be returned.
    """
    if not hasattr(model, "params_") or not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # Calculate residuals
    predictions = model.predict(t)
    residuals = y - predictions

    # Create figure
    n_rows = 2 if acf_only else 3
    if figsize[1] is None:
        figsize = (figsize[0], 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot residuals
    axes[0].plot(t, residuals, color=color_residuals)
    axes[0].axhline(0, linestyle="--", color="k", alpha=0.7)
    axes[0].set_title("Residuals")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual")

    # Plot ACF
    plot_acf(residuals, ax=axes[1], lags=lags, color=color_acf)
    axes[1].set_title("Autocorrelation Function (ACF)")

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def plot_acf_only(
    data: np.ndarray, title: str = "Autocorrelation Function", lags: int = 30
):
    """
    Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def plot_pacf_only(
    data: np.ndarray, title: str = "Partial Autocorrelation Function", lags: int = 30
):
    """
    Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()
