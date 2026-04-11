# src/innovate/plots/diagnostics.py

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    args = [model, t, y, title, lags, acf_only, figsize, color_residuals, color_acf, color_pacf, show]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_residuals__mutmut_orig, x_plot_residuals__mutmut_mutants, args, kwargs, None)


def x_plot_residuals__mutmut_orig(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_1(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "XXResidual AnalysisXX",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_2(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "residual analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_3(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "RESIDUAL ANALYSIS",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_4(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 31,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_5(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = True,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_6(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "XXC0XX",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_7(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "c0",
    color_acf: str = "C1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_8(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "XXC1XX",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_9(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "c1",
    color_pacf: str = "C2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_10(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "XXC2XX",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_11(
    model,
    t: np.ndarray,
    y: np.ndarray,
    title: str = "Residual Analysis",
    lags: int = 30,
    acf_only: bool = False,
    figsize: tuple = (10, None),
    color_residuals: str = "C0",
    color_acf: str = "C1",
    color_pacf: str = "c2",
    show: bool = True,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_12(
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
    show: bool = False,
):
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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


def x_plot_residuals__mutmut_13(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, "params_") and not model.params_:
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


def x_plot_residuals__mutmut_14(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if hasattr(model, "params_") or not model.params_:
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


def x_plot_residuals__mutmut_15(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(None, "params_") or not model.params_:
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


def x_plot_residuals__mutmut_16(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, None) or not model.params_:
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


def x_plot_residuals__mutmut_17(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr("params_") or not model.params_:
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


def x_plot_residuals__mutmut_18(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, ) or not model.params_:
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


def x_plot_residuals__mutmut_19(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, "XXparams_XX") or not model.params_:
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


def x_plot_residuals__mutmut_20(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, "PARAMS_") or not model.params_:
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


def x_plot_residuals__mutmut_21(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if not hasattr(model, "params_") or model.params_:
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


def x_plot_residuals__mutmut_22(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        raise RuntimeError(None)

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


def x_plot_residuals__mutmut_23(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

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


def x_plot_residuals__mutmut_24(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        raise RuntimeError("model has not been fitted yet. call .fit() first.")

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


def x_plot_residuals__mutmut_25(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

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


def x_plot_residuals__mutmut_26(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    predictions = None
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


def x_plot_residuals__mutmut_27(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    predictions = model.predict(None)
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


def x_plot_residuals__mutmut_28(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    residuals = None

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


def x_plot_residuals__mutmut_29(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    residuals = y + predictions

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


def x_plot_residuals__mutmut_30(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    n_rows = None
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


def x_plot_residuals__mutmut_31(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    n_rows = 3 if acf_only else 3
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


def x_plot_residuals__mutmut_32(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    n_rows = 2 if acf_only else 4
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


def x_plot_residuals__mutmut_33(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if figsize[2] is None:
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


def x_plot_residuals__mutmut_34(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    if figsize[1] is not None:
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


def x_plot_residuals__mutmut_35(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        figsize = None

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


def x_plot_residuals__mutmut_36(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        figsize = (figsize[1], 4 * n_rows)

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


def x_plot_residuals__mutmut_37(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        figsize = (figsize[0], 4 / n_rows)

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


def x_plot_residuals__mutmut_38(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        figsize = (figsize[0], 5 * n_rows)

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


def x_plot_residuals__mutmut_39(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = None
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


def x_plot_residuals__mutmut_40(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(None, 1, figsize=figsize)
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


def x_plot_residuals__mutmut_41(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(n_rows, None, figsize=figsize)
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


def x_plot_residuals__mutmut_42(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(n_rows, 1, figsize=None)
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


def x_plot_residuals__mutmut_43(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(1, figsize=figsize)
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


def x_plot_residuals__mutmut_44(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(n_rows, figsize=figsize)
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


def x_plot_residuals__mutmut_45(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(n_rows, 1, )
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


def x_plot_residuals__mutmut_46(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
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


def x_plot_residuals__mutmut_47(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    fig.suptitle(None, fontsize=16)

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


def x_plot_residuals__mutmut_48(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    fig.suptitle(title, fontsize=None)

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


def x_plot_residuals__mutmut_49(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    fig.suptitle(fontsize=16)

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


def x_plot_residuals__mutmut_50(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    fig.suptitle(title, )

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


def x_plot_residuals__mutmut_51(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    fig.suptitle(title, fontsize=17)

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


def x_plot_residuals__mutmut_52(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(None, residuals, color=color_residuals)
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


def x_plot_residuals__mutmut_53(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(t, None, color=color_residuals)
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


def x_plot_residuals__mutmut_54(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(t, residuals, color=None)
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


def x_plot_residuals__mutmut_55(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(residuals, color=color_residuals)
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


def x_plot_residuals__mutmut_56(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(t, color=color_residuals)
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


def x_plot_residuals__mutmut_57(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].plot(t, residuals, )
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


def x_plot_residuals__mutmut_58(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].plot(t, residuals, color=color_residuals)
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


def x_plot_residuals__mutmut_59(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(None, linestyle="--", color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_60(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle=None, color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_61(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color=None, alpha=0.7)
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


def x_plot_residuals__mutmut_62(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color="k", alpha=None)
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


def x_plot_residuals__mutmut_63(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(linestyle="--", color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_64(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_65(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", alpha=0.7)
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


def x_plot_residuals__mutmut_66(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color="k", )
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


def x_plot_residuals__mutmut_67(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].axhline(0, linestyle="--", color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_68(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(1, linestyle="--", color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_69(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="XX--XX", color="k", alpha=0.7)
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


def x_plot_residuals__mutmut_70(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color="XXkXX", alpha=0.7)
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


def x_plot_residuals__mutmut_71(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color="K", alpha=0.7)
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


def x_plot_residuals__mutmut_72(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].axhline(0, linestyle="--", color="k", alpha=1.7)
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


def x_plot_residuals__mutmut_73(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_title(None)
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


def x_plot_residuals__mutmut_74(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_title("Residuals")
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


def x_plot_residuals__mutmut_75(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_title("XXResidualsXX")
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


def x_plot_residuals__mutmut_76(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_title("residuals")
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


def x_plot_residuals__mutmut_77(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_title("RESIDUALS")
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


def x_plot_residuals__mutmut_78(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_xlabel(None)
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


def x_plot_residuals__mutmut_79(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_xlabel("Time")
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


def x_plot_residuals__mutmut_80(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_xlabel("XXTimeXX")
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


def x_plot_residuals__mutmut_81(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_xlabel("time")
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


def x_plot_residuals__mutmut_82(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_xlabel("TIME")
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


def x_plot_residuals__mutmut_83(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_ylabel(None)

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


def x_plot_residuals__mutmut_84(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_ylabel("Residual")

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


def x_plot_residuals__mutmut_85(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_ylabel("XXResidualXX")

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


def x_plot_residuals__mutmut_86(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_ylabel("residual")

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


def x_plot_residuals__mutmut_87(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[0].set_ylabel("RESIDUAL")

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


def x_plot_residuals__mutmut_88(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(None, ax=axes[1], lags=lags, color=color_acf)
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


def x_plot_residuals__mutmut_89(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=None, lags=lags, color=color_acf)
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


def x_plot_residuals__mutmut_90(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=axes[1], lags=None, color=color_acf)
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


def x_plot_residuals__mutmut_91(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=axes[1], lags=lags, color=None)
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


def x_plot_residuals__mutmut_92(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(ax=axes[1], lags=lags, color=color_acf)
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


def x_plot_residuals__mutmut_93(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, lags=lags, color=color_acf)
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


def x_plot_residuals__mutmut_94(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=axes[1], color=color_acf)
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


def x_plot_residuals__mutmut_95(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=axes[1], lags=lags, )
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


def x_plot_residuals__mutmut_96(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    plot_acf(residuals, ax=axes[2], lags=lags, color=color_acf)
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


def x_plot_residuals__mutmut_97(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_title(None)

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_98(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[2].set_title("Autocorrelation Function (ACF)")

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_99(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_title("XXAutocorrelation Function (ACF)XX")

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_100(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_title("autocorrelation function (acf)")

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_101(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
    axes[1].set_title("AUTOCORRELATION FUNCTION (ACF)")

    if not acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_102(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    if acf_only:
        # Plot PACF
        plot_pacf(residuals, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_103(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(None, ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_104(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=None, lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_105(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=axes[2], lags=None, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_106(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=axes[2], lags=lags, color=None)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_107(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(ax=axes[2], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_108(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_109(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=axes[2], color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_110(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=axes[2], lags=lags, )
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_111(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        plot_pacf(residuals, ax=axes[3], lags=lags, color=color_pacf)
        axes[2].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_112(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        axes[2].set_title(None)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_113(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        axes[3].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_114(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        axes[2].set_title("XXPartial Autocorrelation Function (PACF)XX")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_115(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        axes[2].set_title("partial autocorrelation function (pacf)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_116(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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
        axes[2].set_title("PARTIAL AUTOCORRELATION FUNCTION (PACF)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_117(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    plt.tight_layout(rect=None)

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_118(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    plt.tight_layout(rect=[1, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_119(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    plt.tight_layout(rect=[0, 1, 1, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_120(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    plt.tight_layout(rect=[0, 0, 2, 0.96])

    if show:
        plt.show()
    else:
        return fig, axes


def x_plot_residuals__mutmut_121(
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
    """Plots the residuals of a fitted model, along with their ACF and PACF plots.

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

    plt.tight_layout(rect=[0, 0, 1, 1.96])

    if show:
        plt.show()
    else:
        return fig, axes

x_plot_residuals__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_residuals__mutmut_1': x_plot_residuals__mutmut_1, 
    'x_plot_residuals__mutmut_2': x_plot_residuals__mutmut_2, 
    'x_plot_residuals__mutmut_3': x_plot_residuals__mutmut_3, 
    'x_plot_residuals__mutmut_4': x_plot_residuals__mutmut_4, 
    'x_plot_residuals__mutmut_5': x_plot_residuals__mutmut_5, 
    'x_plot_residuals__mutmut_6': x_plot_residuals__mutmut_6, 
    'x_plot_residuals__mutmut_7': x_plot_residuals__mutmut_7, 
    'x_plot_residuals__mutmut_8': x_plot_residuals__mutmut_8, 
    'x_plot_residuals__mutmut_9': x_plot_residuals__mutmut_9, 
    'x_plot_residuals__mutmut_10': x_plot_residuals__mutmut_10, 
    'x_plot_residuals__mutmut_11': x_plot_residuals__mutmut_11, 
    'x_plot_residuals__mutmut_12': x_plot_residuals__mutmut_12, 
    'x_plot_residuals__mutmut_13': x_plot_residuals__mutmut_13, 
    'x_plot_residuals__mutmut_14': x_plot_residuals__mutmut_14, 
    'x_plot_residuals__mutmut_15': x_plot_residuals__mutmut_15, 
    'x_plot_residuals__mutmut_16': x_plot_residuals__mutmut_16, 
    'x_plot_residuals__mutmut_17': x_plot_residuals__mutmut_17, 
    'x_plot_residuals__mutmut_18': x_plot_residuals__mutmut_18, 
    'x_plot_residuals__mutmut_19': x_plot_residuals__mutmut_19, 
    'x_plot_residuals__mutmut_20': x_plot_residuals__mutmut_20, 
    'x_plot_residuals__mutmut_21': x_plot_residuals__mutmut_21, 
    'x_plot_residuals__mutmut_22': x_plot_residuals__mutmut_22, 
    'x_plot_residuals__mutmut_23': x_plot_residuals__mutmut_23, 
    'x_plot_residuals__mutmut_24': x_plot_residuals__mutmut_24, 
    'x_plot_residuals__mutmut_25': x_plot_residuals__mutmut_25, 
    'x_plot_residuals__mutmut_26': x_plot_residuals__mutmut_26, 
    'x_plot_residuals__mutmut_27': x_plot_residuals__mutmut_27, 
    'x_plot_residuals__mutmut_28': x_plot_residuals__mutmut_28, 
    'x_plot_residuals__mutmut_29': x_plot_residuals__mutmut_29, 
    'x_plot_residuals__mutmut_30': x_plot_residuals__mutmut_30, 
    'x_plot_residuals__mutmut_31': x_plot_residuals__mutmut_31, 
    'x_plot_residuals__mutmut_32': x_plot_residuals__mutmut_32, 
    'x_plot_residuals__mutmut_33': x_plot_residuals__mutmut_33, 
    'x_plot_residuals__mutmut_34': x_plot_residuals__mutmut_34, 
    'x_plot_residuals__mutmut_35': x_plot_residuals__mutmut_35, 
    'x_plot_residuals__mutmut_36': x_plot_residuals__mutmut_36, 
    'x_plot_residuals__mutmut_37': x_plot_residuals__mutmut_37, 
    'x_plot_residuals__mutmut_38': x_plot_residuals__mutmut_38, 
    'x_plot_residuals__mutmut_39': x_plot_residuals__mutmut_39, 
    'x_plot_residuals__mutmut_40': x_plot_residuals__mutmut_40, 
    'x_plot_residuals__mutmut_41': x_plot_residuals__mutmut_41, 
    'x_plot_residuals__mutmut_42': x_plot_residuals__mutmut_42, 
    'x_plot_residuals__mutmut_43': x_plot_residuals__mutmut_43, 
    'x_plot_residuals__mutmut_44': x_plot_residuals__mutmut_44, 
    'x_plot_residuals__mutmut_45': x_plot_residuals__mutmut_45, 
    'x_plot_residuals__mutmut_46': x_plot_residuals__mutmut_46, 
    'x_plot_residuals__mutmut_47': x_plot_residuals__mutmut_47, 
    'x_plot_residuals__mutmut_48': x_plot_residuals__mutmut_48, 
    'x_plot_residuals__mutmut_49': x_plot_residuals__mutmut_49, 
    'x_plot_residuals__mutmut_50': x_plot_residuals__mutmut_50, 
    'x_plot_residuals__mutmut_51': x_plot_residuals__mutmut_51, 
    'x_plot_residuals__mutmut_52': x_plot_residuals__mutmut_52, 
    'x_plot_residuals__mutmut_53': x_plot_residuals__mutmut_53, 
    'x_plot_residuals__mutmut_54': x_plot_residuals__mutmut_54, 
    'x_plot_residuals__mutmut_55': x_plot_residuals__mutmut_55, 
    'x_plot_residuals__mutmut_56': x_plot_residuals__mutmut_56, 
    'x_plot_residuals__mutmut_57': x_plot_residuals__mutmut_57, 
    'x_plot_residuals__mutmut_58': x_plot_residuals__mutmut_58, 
    'x_plot_residuals__mutmut_59': x_plot_residuals__mutmut_59, 
    'x_plot_residuals__mutmut_60': x_plot_residuals__mutmut_60, 
    'x_plot_residuals__mutmut_61': x_plot_residuals__mutmut_61, 
    'x_plot_residuals__mutmut_62': x_plot_residuals__mutmut_62, 
    'x_plot_residuals__mutmut_63': x_plot_residuals__mutmut_63, 
    'x_plot_residuals__mutmut_64': x_plot_residuals__mutmut_64, 
    'x_plot_residuals__mutmut_65': x_plot_residuals__mutmut_65, 
    'x_plot_residuals__mutmut_66': x_plot_residuals__mutmut_66, 
    'x_plot_residuals__mutmut_67': x_plot_residuals__mutmut_67, 
    'x_plot_residuals__mutmut_68': x_plot_residuals__mutmut_68, 
    'x_plot_residuals__mutmut_69': x_plot_residuals__mutmut_69, 
    'x_plot_residuals__mutmut_70': x_plot_residuals__mutmut_70, 
    'x_plot_residuals__mutmut_71': x_plot_residuals__mutmut_71, 
    'x_plot_residuals__mutmut_72': x_plot_residuals__mutmut_72, 
    'x_plot_residuals__mutmut_73': x_plot_residuals__mutmut_73, 
    'x_plot_residuals__mutmut_74': x_plot_residuals__mutmut_74, 
    'x_plot_residuals__mutmut_75': x_plot_residuals__mutmut_75, 
    'x_plot_residuals__mutmut_76': x_plot_residuals__mutmut_76, 
    'x_plot_residuals__mutmut_77': x_plot_residuals__mutmut_77, 
    'x_plot_residuals__mutmut_78': x_plot_residuals__mutmut_78, 
    'x_plot_residuals__mutmut_79': x_plot_residuals__mutmut_79, 
    'x_plot_residuals__mutmut_80': x_plot_residuals__mutmut_80, 
    'x_plot_residuals__mutmut_81': x_plot_residuals__mutmut_81, 
    'x_plot_residuals__mutmut_82': x_plot_residuals__mutmut_82, 
    'x_plot_residuals__mutmut_83': x_plot_residuals__mutmut_83, 
    'x_plot_residuals__mutmut_84': x_plot_residuals__mutmut_84, 
    'x_plot_residuals__mutmut_85': x_plot_residuals__mutmut_85, 
    'x_plot_residuals__mutmut_86': x_plot_residuals__mutmut_86, 
    'x_plot_residuals__mutmut_87': x_plot_residuals__mutmut_87, 
    'x_plot_residuals__mutmut_88': x_plot_residuals__mutmut_88, 
    'x_plot_residuals__mutmut_89': x_plot_residuals__mutmut_89, 
    'x_plot_residuals__mutmut_90': x_plot_residuals__mutmut_90, 
    'x_plot_residuals__mutmut_91': x_plot_residuals__mutmut_91, 
    'x_plot_residuals__mutmut_92': x_plot_residuals__mutmut_92, 
    'x_plot_residuals__mutmut_93': x_plot_residuals__mutmut_93, 
    'x_plot_residuals__mutmut_94': x_plot_residuals__mutmut_94, 
    'x_plot_residuals__mutmut_95': x_plot_residuals__mutmut_95, 
    'x_plot_residuals__mutmut_96': x_plot_residuals__mutmut_96, 
    'x_plot_residuals__mutmut_97': x_plot_residuals__mutmut_97, 
    'x_plot_residuals__mutmut_98': x_plot_residuals__mutmut_98, 
    'x_plot_residuals__mutmut_99': x_plot_residuals__mutmut_99, 
    'x_plot_residuals__mutmut_100': x_plot_residuals__mutmut_100, 
    'x_plot_residuals__mutmut_101': x_plot_residuals__mutmut_101, 
    'x_plot_residuals__mutmut_102': x_plot_residuals__mutmut_102, 
    'x_plot_residuals__mutmut_103': x_plot_residuals__mutmut_103, 
    'x_plot_residuals__mutmut_104': x_plot_residuals__mutmut_104, 
    'x_plot_residuals__mutmut_105': x_plot_residuals__mutmut_105, 
    'x_plot_residuals__mutmut_106': x_plot_residuals__mutmut_106, 
    'x_plot_residuals__mutmut_107': x_plot_residuals__mutmut_107, 
    'x_plot_residuals__mutmut_108': x_plot_residuals__mutmut_108, 
    'x_plot_residuals__mutmut_109': x_plot_residuals__mutmut_109, 
    'x_plot_residuals__mutmut_110': x_plot_residuals__mutmut_110, 
    'x_plot_residuals__mutmut_111': x_plot_residuals__mutmut_111, 
    'x_plot_residuals__mutmut_112': x_plot_residuals__mutmut_112, 
    'x_plot_residuals__mutmut_113': x_plot_residuals__mutmut_113, 
    'x_plot_residuals__mutmut_114': x_plot_residuals__mutmut_114, 
    'x_plot_residuals__mutmut_115': x_plot_residuals__mutmut_115, 
    'x_plot_residuals__mutmut_116': x_plot_residuals__mutmut_116, 
    'x_plot_residuals__mutmut_117': x_plot_residuals__mutmut_117, 
    'x_plot_residuals__mutmut_118': x_plot_residuals__mutmut_118, 
    'x_plot_residuals__mutmut_119': x_plot_residuals__mutmut_119, 
    'x_plot_residuals__mutmut_120': x_plot_residuals__mutmut_120, 
    'x_plot_residuals__mutmut_121': x_plot_residuals__mutmut_121
}
x_plot_residuals__mutmut_orig.__name__ = 'x_plot_residuals'


def plot_acf_only(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    args = [data, title, lags]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_acf_only__mutmut_orig, x_plot_acf_only__mutmut_mutants, args, kwargs, None)


def x_plot_acf_only__mutmut_orig(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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


def x_plot_acf_only__mutmut_1(
    data: np.ndarray,
    title: str = "XXAutocorrelation FunctionXX",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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


def x_plot_acf_only__mutmut_2(
    data: np.ndarray,
    title: str = "autocorrelation function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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


def x_plot_acf_only__mutmut_3(
    data: np.ndarray,
    title: str = "AUTOCORRELATION FUNCTION",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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


def x_plot_acf_only__mutmut_4(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 31,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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


def x_plot_acf_only__mutmut_5(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = None
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_6(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(None, 1, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_7(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, None, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_8(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=None)
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_9(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_10(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_11(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, )
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_12(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_13(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_14(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_15(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the ACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_acf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_16(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(None, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_17(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(data, ax=None, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_18(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(data, ax=ax, lags=None)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_19(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_20(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(data, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_21(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    plot_acf(data, ax=ax, )
    ax.set_title(title)
    plt.show()


def x_plot_acf_only__mutmut_22(
    data: np.ndarray,
    title: str = "Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Autocorrelation Function (ACF) of a time series.

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
    ax.set_title(None)
    plt.show()

x_plot_acf_only__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_acf_only__mutmut_1': x_plot_acf_only__mutmut_1, 
    'x_plot_acf_only__mutmut_2': x_plot_acf_only__mutmut_2, 
    'x_plot_acf_only__mutmut_3': x_plot_acf_only__mutmut_3, 
    'x_plot_acf_only__mutmut_4': x_plot_acf_only__mutmut_4, 
    'x_plot_acf_only__mutmut_5': x_plot_acf_only__mutmut_5, 
    'x_plot_acf_only__mutmut_6': x_plot_acf_only__mutmut_6, 
    'x_plot_acf_only__mutmut_7': x_plot_acf_only__mutmut_7, 
    'x_plot_acf_only__mutmut_8': x_plot_acf_only__mutmut_8, 
    'x_plot_acf_only__mutmut_9': x_plot_acf_only__mutmut_9, 
    'x_plot_acf_only__mutmut_10': x_plot_acf_only__mutmut_10, 
    'x_plot_acf_only__mutmut_11': x_plot_acf_only__mutmut_11, 
    'x_plot_acf_only__mutmut_12': x_plot_acf_only__mutmut_12, 
    'x_plot_acf_only__mutmut_13': x_plot_acf_only__mutmut_13, 
    'x_plot_acf_only__mutmut_14': x_plot_acf_only__mutmut_14, 
    'x_plot_acf_only__mutmut_15': x_plot_acf_only__mutmut_15, 
    'x_plot_acf_only__mutmut_16': x_plot_acf_only__mutmut_16, 
    'x_plot_acf_only__mutmut_17': x_plot_acf_only__mutmut_17, 
    'x_plot_acf_only__mutmut_18': x_plot_acf_only__mutmut_18, 
    'x_plot_acf_only__mutmut_19': x_plot_acf_only__mutmut_19, 
    'x_plot_acf_only__mutmut_20': x_plot_acf_only__mutmut_20, 
    'x_plot_acf_only__mutmut_21': x_plot_acf_only__mutmut_21, 
    'x_plot_acf_only__mutmut_22': x_plot_acf_only__mutmut_22
}
x_plot_acf_only__mutmut_orig.__name__ = 'x_plot_acf_only'


def plot_pacf_only(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    args = [data, title, lags]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_pacf_only__mutmut_orig, x_plot_pacf_only__mutmut_mutants, args, kwargs, None)


def x_plot_pacf_only__mutmut_orig(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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


def x_plot_pacf_only__mutmut_1(
    data: np.ndarray,
    title: str = "XXPartial Autocorrelation FunctionXX",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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


def x_plot_pacf_only__mutmut_2(
    data: np.ndarray,
    title: str = "partial autocorrelation function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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


def x_plot_pacf_only__mutmut_3(
    data: np.ndarray,
    title: str = "PARTIAL AUTOCORRELATION FUNCTION",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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


def x_plot_pacf_only__mutmut_4(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 31,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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


def x_plot_pacf_only__mutmut_5(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = None
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_6(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(None, 1, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_7(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, None, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_8(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=None)
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_9(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_10(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_11(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, )
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_12(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_13(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_14(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_15(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

    Parameters
    ----------
    data : np.ndarray
        The time series data.
    title : str, optional
        The title for the plot, by default "Partial Autocorrelation Function".
    lags : int, optional
        The number of lags to show in the PACF plot, by default 30.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_pacf(data, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_16(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(None, ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_17(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(data, ax=None, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_18(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(data, ax=ax, lags=None)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_19(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(ax=ax, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_20(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(data, lags=lags)
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_21(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    plot_pacf(data, ax=ax, )
    ax.set_title(title)
    plt.show()


def x_plot_pacf_only__mutmut_22(
    data: np.ndarray,
    title: str = "Partial Autocorrelation Function",
    lags: int = 30,
):
    """Plots the Partial Autocorrelation Function (PACF) of a time series.

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
    ax.set_title(None)
    plt.show()

x_plot_pacf_only__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_pacf_only__mutmut_1': x_plot_pacf_only__mutmut_1, 
    'x_plot_pacf_only__mutmut_2': x_plot_pacf_only__mutmut_2, 
    'x_plot_pacf_only__mutmut_3': x_plot_pacf_only__mutmut_3, 
    'x_plot_pacf_only__mutmut_4': x_plot_pacf_only__mutmut_4, 
    'x_plot_pacf_only__mutmut_5': x_plot_pacf_only__mutmut_5, 
    'x_plot_pacf_only__mutmut_6': x_plot_pacf_only__mutmut_6, 
    'x_plot_pacf_only__mutmut_7': x_plot_pacf_only__mutmut_7, 
    'x_plot_pacf_only__mutmut_8': x_plot_pacf_only__mutmut_8, 
    'x_plot_pacf_only__mutmut_9': x_plot_pacf_only__mutmut_9, 
    'x_plot_pacf_only__mutmut_10': x_plot_pacf_only__mutmut_10, 
    'x_plot_pacf_only__mutmut_11': x_plot_pacf_only__mutmut_11, 
    'x_plot_pacf_only__mutmut_12': x_plot_pacf_only__mutmut_12, 
    'x_plot_pacf_only__mutmut_13': x_plot_pacf_only__mutmut_13, 
    'x_plot_pacf_only__mutmut_14': x_plot_pacf_only__mutmut_14, 
    'x_plot_pacf_only__mutmut_15': x_plot_pacf_only__mutmut_15, 
    'x_plot_pacf_only__mutmut_16': x_plot_pacf_only__mutmut_16, 
    'x_plot_pacf_only__mutmut_17': x_plot_pacf_only__mutmut_17, 
    'x_plot_pacf_only__mutmut_18': x_plot_pacf_only__mutmut_18, 
    'x_plot_pacf_only__mutmut_19': x_plot_pacf_only__mutmut_19, 
    'x_plot_pacf_only__mutmut_20': x_plot_pacf_only__mutmut_20, 
    'x_plot_pacf_only__mutmut_21': x_plot_pacf_only__mutmut_21, 
    'x_plot_pacf_only__mutmut_22': x_plot_pacf_only__mutmut_22
}
x_plot_pacf_only__mutmut_orig.__name__ = 'x_plot_pacf_only'
