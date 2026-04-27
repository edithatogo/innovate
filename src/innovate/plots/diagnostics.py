"""Diagnostic plotting helpers for fitted diffusion models."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from innovate.fitters.diagnostics_contract import DiagnosticsContract
from innovate.fitters.residual_analysis import ResidualAnalysis


def _extract_diagnostics_payload(
    diagnostics: object | None,
) -> tuple[np.ndarray | None, ResidualAnalysis | None, dict[str, object]]:
    """Pull residuals and metadata from a diagnostics object when available."""
    if diagnostics is None:
        return None, None, {}

    residuals = getattr(diagnostics, "residuals", None)
    residual_analysis = getattr(diagnostics, "residual_analysis", None)

    metadata: dict[str, object] = {}
    for key in ("support_level", "provenance", "comparison_family", "model_name", "uncertainty"):
        if hasattr(diagnostics, key):
            value = getattr(diagnostics, key)
            if key == "uncertainty" and hasattr(value, "to_dict"):
                metadata[key] = value.to_dict()
            elif is_dataclass(value):
                metadata[key] = asdict(value)
            else:
                metadata[key] = value

    return (
        None if residuals is None else np.asarray(residuals, dtype=float),
        residual_analysis if isinstance(residual_analysis, ResidualAnalysis) else None,
        metadata,
    )


def plot_residuals(  # noqa: PLR0912, PLR0915
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
    diagnostics: DiagnosticsContract | object | None = None,
    residual_analysis: ResidualAnalysis | None = None,
):
    """Plot residuals, ACF, and PACF for a fitted model.

    Parameters
    ----------
        model : innovate.base.base.DiffusionModel
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
        diagnostics : innovate.fitters.diagnostics_contract.DiagnosticsContract | object | None, optional
        Optional diagnostics contract or fit diagnostics object. If provided,
        the function reuses the stored residuals and residual analysis instead of
        recomputing them.
    residual_analysis : ResidualAnalysis | None, optional
        Explicit residual analysis object to plot. Takes precedence over the
        analysis attached to ``diagnostics``.
    """
    if not hasattr(model, "params_") or not model.params_:
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    diagnostics_residuals, diagnostics_analysis, diagnostics_metadata = _extract_diagnostics_payload(diagnostics)
    if residual_analysis is None:
        residual_analysis = diagnostics_analysis

    if diagnostics_residuals is not None:
        residuals = diagnostics_residuals
    else:
        predictions = np.asarray(model.predict(t_arr), dtype=float)
        residuals = y_arr - predictions

    if residual_analysis is None and diagnostics is not None and hasattr(diagnostics, "residual_analysis"):
        residual_analysis = diagnostics.residual_analysis
        if not isinstance(residual_analysis, ResidualAnalysis):
            residual_analysis = None

    # Create figure
    n_rows = 2 if acf_only else 3
    if figsize[1] is None:
        figsize = (figsize[0], 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot residuals
    axes[0].plot(t_arr, residuals, color=color_residuals)
    axes[0].axhline(0, linestyle="--", color="k", alpha=0.7)
    axes[0].set_title("Residuals")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual")

    if diagnostics_metadata:
        info_bits = []
        support_level = diagnostics_metadata.get("support_level")
        provenance = diagnostics_metadata.get("provenance")
        uncertainty = diagnostics_metadata.get("uncertainty")
        if support_level:
            info_bits.append(f"support={support_level}")
        if provenance:
            info_bits.append(f"provenance={provenance}")
        if isinstance(uncertainty, dict):
            report_type = uncertainty.get("report_type")
            if report_type:
                info_bits.append(f"uncertainty={report_type}")
        if info_bits:
            axes[0].text(
                0.01,
                0.98,
                " | ".join(str(bit) for bit in info_bits),
                transform=axes[0].transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )

    acf_lags = min(int(lags), max(1, len(residuals) - 1))

    # Plot ACF
    plot_acf(residuals, ax=axes[1], lags=acf_lags, color=color_acf)
    axes[1].set_title("Autocorrelation Function (ACF)")

    if not acf_only:
        pacf_lags = min(int(lags), max(1, len(residuals) // 2))
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=pacf_lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes
    return None


def plot_acf_only(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plot the autocorrelation function of a time series.

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
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plot the partial autocorrelation function of a time series.

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
