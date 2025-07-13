# src/innovate/plots/diagnostics.py

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

def plot_residuals(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30
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
    """
    if not hasattr(model, 'params_') or not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    # Calculate residuals
    predictions = model.predict(t)
    residuals = y - predictions

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=16)

    # Plot residuals
    axes[0].plot(t, residuals)
    axes[0].axhline(0, linestyle='--', color='k', alpha=0.7)
    axes[0].set_title("Residuals")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual")

    # Plot ACF
    plot_acf(residuals, ax=axes[1], lags=lags)
    axes[1].set_title("Autocorrelation Function (ACF)")

    # Plot PACF
    plot_pacf(residuals, ax=axes[2], lags=lags)
    axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_acf_only(data: np.ndarray, title: str = "Autocorrelation Function", lags: int = 30):
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
