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


def plot_diffusion_curve(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    args = [t, y_obs, y_pred, title, xlabel, ylabel, save_path]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_diffusion_curve__mutmut_orig, x_plot_diffusion_curve__mutmut_mutants, args, kwargs, None)


def x_plot_diffusion_curve__mutmut_orig(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_1(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "XXDiffusion CurveXX",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_2(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "diffusion curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_3(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "DIFFUSION CURVE",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_4(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "XXTimeXX",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_5(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_6(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "TIME",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_7(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "XXCumulative AdoptionsXX",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_8(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "cumulative adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_9(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "CUMULATIVE ADOPTIONS",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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


def x_plot_diffusion_curve__mutmut_10(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=None)
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


def x_plot_diffusion_curve__mutmut_11(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(11, 6))
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


def x_plot_diffusion_curve__mutmut_12(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 7))
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


def x_plot_diffusion_curve__mutmut_13(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(None, y_obs, "o", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_14(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, None, "o", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_15(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, None, label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_16(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label=None, alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_17(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="Observed", alpha=None)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_18(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_obs, "o", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_19(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, "o", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_20(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_21(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_22(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="Observed", )
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_23(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "XXoXX", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_24(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "O", label="Observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_25(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="XXObservedXX", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_26(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="observed", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_27(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="OBSERVED", alpha=0.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_28(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
        t: Time points.
        y_obs: Observed cumulative adoptions.
        y_pred: Predicted cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_obs, "o", label="Observed", alpha=1.6)
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_29(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(None, y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_30(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, None, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_31(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, None, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_32(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label=None, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_33(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_34(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(y_pred, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_35(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, "-", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_36(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_37(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_38(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="Predicted", )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_39(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "XX-XX", label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_40(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="XXPredictedXX", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_41(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="predicted", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_42(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="PREDICTED", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_43(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.plot(t, y_pred, "-", label="Predicted", linewidth=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_44(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.title(None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_45(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.xlabel(None)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_46(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.ylabel(None)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_47(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.grid(None)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_48(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
    plt.grid(False)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_diffusion_curve__mutmut_49(
    t: Sequence[float],
    y_obs: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Diffusion Curve",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves.

    Args:
    ----
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
        plt.savefig(None)
    plt.show()

x_plot_diffusion_curve__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_diffusion_curve__mutmut_1': x_plot_diffusion_curve__mutmut_1, 
    'x_plot_diffusion_curve__mutmut_2': x_plot_diffusion_curve__mutmut_2, 
    'x_plot_diffusion_curve__mutmut_3': x_plot_diffusion_curve__mutmut_3, 
    'x_plot_diffusion_curve__mutmut_4': x_plot_diffusion_curve__mutmut_4, 
    'x_plot_diffusion_curve__mutmut_5': x_plot_diffusion_curve__mutmut_5, 
    'x_plot_diffusion_curve__mutmut_6': x_plot_diffusion_curve__mutmut_6, 
    'x_plot_diffusion_curve__mutmut_7': x_plot_diffusion_curve__mutmut_7, 
    'x_plot_diffusion_curve__mutmut_8': x_plot_diffusion_curve__mutmut_8, 
    'x_plot_diffusion_curve__mutmut_9': x_plot_diffusion_curve__mutmut_9, 
    'x_plot_diffusion_curve__mutmut_10': x_plot_diffusion_curve__mutmut_10, 
    'x_plot_diffusion_curve__mutmut_11': x_plot_diffusion_curve__mutmut_11, 
    'x_plot_diffusion_curve__mutmut_12': x_plot_diffusion_curve__mutmut_12, 
    'x_plot_diffusion_curve__mutmut_13': x_plot_diffusion_curve__mutmut_13, 
    'x_plot_diffusion_curve__mutmut_14': x_plot_diffusion_curve__mutmut_14, 
    'x_plot_diffusion_curve__mutmut_15': x_plot_diffusion_curve__mutmut_15, 
    'x_plot_diffusion_curve__mutmut_16': x_plot_diffusion_curve__mutmut_16, 
    'x_plot_diffusion_curve__mutmut_17': x_plot_diffusion_curve__mutmut_17, 
    'x_plot_diffusion_curve__mutmut_18': x_plot_diffusion_curve__mutmut_18, 
    'x_plot_diffusion_curve__mutmut_19': x_plot_diffusion_curve__mutmut_19, 
    'x_plot_diffusion_curve__mutmut_20': x_plot_diffusion_curve__mutmut_20, 
    'x_plot_diffusion_curve__mutmut_21': x_plot_diffusion_curve__mutmut_21, 
    'x_plot_diffusion_curve__mutmut_22': x_plot_diffusion_curve__mutmut_22, 
    'x_plot_diffusion_curve__mutmut_23': x_plot_diffusion_curve__mutmut_23, 
    'x_plot_diffusion_curve__mutmut_24': x_plot_diffusion_curve__mutmut_24, 
    'x_plot_diffusion_curve__mutmut_25': x_plot_diffusion_curve__mutmut_25, 
    'x_plot_diffusion_curve__mutmut_26': x_plot_diffusion_curve__mutmut_26, 
    'x_plot_diffusion_curve__mutmut_27': x_plot_diffusion_curve__mutmut_27, 
    'x_plot_diffusion_curve__mutmut_28': x_plot_diffusion_curve__mutmut_28, 
    'x_plot_diffusion_curve__mutmut_29': x_plot_diffusion_curve__mutmut_29, 
    'x_plot_diffusion_curve__mutmut_30': x_plot_diffusion_curve__mutmut_30, 
    'x_plot_diffusion_curve__mutmut_31': x_plot_diffusion_curve__mutmut_31, 
    'x_plot_diffusion_curve__mutmut_32': x_plot_diffusion_curve__mutmut_32, 
    'x_plot_diffusion_curve__mutmut_33': x_plot_diffusion_curve__mutmut_33, 
    'x_plot_diffusion_curve__mutmut_34': x_plot_diffusion_curve__mutmut_34, 
    'x_plot_diffusion_curve__mutmut_35': x_plot_diffusion_curve__mutmut_35, 
    'x_plot_diffusion_curve__mutmut_36': x_plot_diffusion_curve__mutmut_36, 
    'x_plot_diffusion_curve__mutmut_37': x_plot_diffusion_curve__mutmut_37, 
    'x_plot_diffusion_curve__mutmut_38': x_plot_diffusion_curve__mutmut_38, 
    'x_plot_diffusion_curve__mutmut_39': x_plot_diffusion_curve__mutmut_39, 
    'x_plot_diffusion_curve__mutmut_40': x_plot_diffusion_curve__mutmut_40, 
    'x_plot_diffusion_curve__mutmut_41': x_plot_diffusion_curve__mutmut_41, 
    'x_plot_diffusion_curve__mutmut_42': x_plot_diffusion_curve__mutmut_42, 
    'x_plot_diffusion_curve__mutmut_43': x_plot_diffusion_curve__mutmut_43, 
    'x_plot_diffusion_curve__mutmut_44': x_plot_diffusion_curve__mutmut_44, 
    'x_plot_diffusion_curve__mutmut_45': x_plot_diffusion_curve__mutmut_45, 
    'x_plot_diffusion_curve__mutmut_46': x_plot_diffusion_curve__mutmut_46, 
    'x_plot_diffusion_curve__mutmut_47': x_plot_diffusion_curve__mutmut_47, 
    'x_plot_diffusion_curve__mutmut_48': x_plot_diffusion_curve__mutmut_48, 
    'x_plot_diffusion_curve__mutmut_49': x_plot_diffusion_curve__mutmut_49
}
x_plot_diffusion_curve__mutmut_orig.__name__ = 'x_plot_diffusion_curve'


def plot_multi_product_diffusion(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    args = [df_pred, df_obs, title, xlabel, ylabel, save_path]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_plot_multi_product_diffusion__mutmut_orig, x_plot_multi_product_diffusion__mutmut_mutants, args, kwargs, None)


def x_plot_multi_product_diffusion__mutmut_orig(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_1(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "XXMulti-Product Diffusion CurvesXX",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_2(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "multi-product diffusion curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_3(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "MULTI-PRODUCT DIFFUSION CURVES",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_4(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "XXTimeXX",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_5(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_6(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "TIME",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_7(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "XXCumulative AdoptionsXX",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_8(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "cumulative adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_9(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "CUMULATIVE ADOPTIONS",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_10(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
        df_pred: DataFrame of predicted cumulative adoptions (index is time, columns are product names).
        df_obs: Optional DataFrame of observed cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=None)

    # Plot predicted curves
    for col in df_pred.columns:
        plt.plot(
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_11(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
        df_pred: DataFrame of predicted cumulative adoptions (index is time, columns are product names).
        df_obs: Optional DataFrame of observed cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(13, 7))

    # Plot predicted curves
    for col in df_pred.columns:
        plt.plot(
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_12(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
        df_pred: DataFrame of predicted cumulative adoptions (index is time, columns are product names).
        df_obs: Optional DataFrame of observed cumulative adoptions.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save the plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(12, 8))

    # Plot predicted curves
    for col in df_pred.columns:
        plt.plot(
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_13(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            None,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_14(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            None,
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_15(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            None,
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_16(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=None,
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_17(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=None,
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


def x_plot_multi_product_diffusion__mutmut_18(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_19(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_20(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_21(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_22(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
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


def x_plot_multi_product_diffusion__mutmut_23(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "XX-XX",
            label=f"Predicted {col}",
            linewidth=2,
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


def x_plot_multi_product_diffusion__mutmut_24(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=3,
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


def x_plot_multi_product_diffusion__mutmut_25(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is None:
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


def x_plot_multi_product_diffusion__mutmut_26(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(None, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_27(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, None, "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_28(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], None, label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_29(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=None, alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_30(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=None)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_31(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_32(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_33(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_34(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_35(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_36(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "XXoXX", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_37(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "O", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_38(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=1.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_39(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_40(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(None)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_41(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(None)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_42(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(None)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_43(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
        )

    # Plot observed data if provided
    if df_obs is not None:
        for col in df_obs.columns:
            plt.plot(df_obs.index, df_obs[col], "o", label=f"Observed {col}", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(False)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def x_plot_multi_product_diffusion__mutmut_44(
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame | None = None,
    title: str = "Multi-Product Diffusion Curves",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Adoptions",
    save_path: str | None = None,
):
    """:no-index:

    Plots observed and predicted diffusion curves for multiple products.

    Args:
    ----
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
            df_pred.index,
            df_pred[col],
            "-",
            label=f"Predicted {col}",
            linewidth=2,
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
        plt.savefig(None)
    plt.show()

x_plot_multi_product_diffusion__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_plot_multi_product_diffusion__mutmut_1': x_plot_multi_product_diffusion__mutmut_1, 
    'x_plot_multi_product_diffusion__mutmut_2': x_plot_multi_product_diffusion__mutmut_2, 
    'x_plot_multi_product_diffusion__mutmut_3': x_plot_multi_product_diffusion__mutmut_3, 
    'x_plot_multi_product_diffusion__mutmut_4': x_plot_multi_product_diffusion__mutmut_4, 
    'x_plot_multi_product_diffusion__mutmut_5': x_plot_multi_product_diffusion__mutmut_5, 
    'x_plot_multi_product_diffusion__mutmut_6': x_plot_multi_product_diffusion__mutmut_6, 
    'x_plot_multi_product_diffusion__mutmut_7': x_plot_multi_product_diffusion__mutmut_7, 
    'x_plot_multi_product_diffusion__mutmut_8': x_plot_multi_product_diffusion__mutmut_8, 
    'x_plot_multi_product_diffusion__mutmut_9': x_plot_multi_product_diffusion__mutmut_9, 
    'x_plot_multi_product_diffusion__mutmut_10': x_plot_multi_product_diffusion__mutmut_10, 
    'x_plot_multi_product_diffusion__mutmut_11': x_plot_multi_product_diffusion__mutmut_11, 
    'x_plot_multi_product_diffusion__mutmut_12': x_plot_multi_product_diffusion__mutmut_12, 
    'x_plot_multi_product_diffusion__mutmut_13': x_plot_multi_product_diffusion__mutmut_13, 
    'x_plot_multi_product_diffusion__mutmut_14': x_plot_multi_product_diffusion__mutmut_14, 
    'x_plot_multi_product_diffusion__mutmut_15': x_plot_multi_product_diffusion__mutmut_15, 
    'x_plot_multi_product_diffusion__mutmut_16': x_plot_multi_product_diffusion__mutmut_16, 
    'x_plot_multi_product_diffusion__mutmut_17': x_plot_multi_product_diffusion__mutmut_17, 
    'x_plot_multi_product_diffusion__mutmut_18': x_plot_multi_product_diffusion__mutmut_18, 
    'x_plot_multi_product_diffusion__mutmut_19': x_plot_multi_product_diffusion__mutmut_19, 
    'x_plot_multi_product_diffusion__mutmut_20': x_plot_multi_product_diffusion__mutmut_20, 
    'x_plot_multi_product_diffusion__mutmut_21': x_plot_multi_product_diffusion__mutmut_21, 
    'x_plot_multi_product_diffusion__mutmut_22': x_plot_multi_product_diffusion__mutmut_22, 
    'x_plot_multi_product_diffusion__mutmut_23': x_plot_multi_product_diffusion__mutmut_23, 
    'x_plot_multi_product_diffusion__mutmut_24': x_plot_multi_product_diffusion__mutmut_24, 
    'x_plot_multi_product_diffusion__mutmut_25': x_plot_multi_product_diffusion__mutmut_25, 
    'x_plot_multi_product_diffusion__mutmut_26': x_plot_multi_product_diffusion__mutmut_26, 
    'x_plot_multi_product_diffusion__mutmut_27': x_plot_multi_product_diffusion__mutmut_27, 
    'x_plot_multi_product_diffusion__mutmut_28': x_plot_multi_product_diffusion__mutmut_28, 
    'x_plot_multi_product_diffusion__mutmut_29': x_plot_multi_product_diffusion__mutmut_29, 
    'x_plot_multi_product_diffusion__mutmut_30': x_plot_multi_product_diffusion__mutmut_30, 
    'x_plot_multi_product_diffusion__mutmut_31': x_plot_multi_product_diffusion__mutmut_31, 
    'x_plot_multi_product_diffusion__mutmut_32': x_plot_multi_product_diffusion__mutmut_32, 
    'x_plot_multi_product_diffusion__mutmut_33': x_plot_multi_product_diffusion__mutmut_33, 
    'x_plot_multi_product_diffusion__mutmut_34': x_plot_multi_product_diffusion__mutmut_34, 
    'x_plot_multi_product_diffusion__mutmut_35': x_plot_multi_product_diffusion__mutmut_35, 
    'x_plot_multi_product_diffusion__mutmut_36': x_plot_multi_product_diffusion__mutmut_36, 
    'x_plot_multi_product_diffusion__mutmut_37': x_plot_multi_product_diffusion__mutmut_37, 
    'x_plot_multi_product_diffusion__mutmut_38': x_plot_multi_product_diffusion__mutmut_38, 
    'x_plot_multi_product_diffusion__mutmut_39': x_plot_multi_product_diffusion__mutmut_39, 
    'x_plot_multi_product_diffusion__mutmut_40': x_plot_multi_product_diffusion__mutmut_40, 
    'x_plot_multi_product_diffusion__mutmut_41': x_plot_multi_product_diffusion__mutmut_41, 
    'x_plot_multi_product_diffusion__mutmut_42': x_plot_multi_product_diffusion__mutmut_42, 
    'x_plot_multi_product_diffusion__mutmut_43': x_plot_multi_product_diffusion__mutmut_43, 
    'x_plot_multi_product_diffusion__mutmut_44': x_plot_multi_product_diffusion__mutmut_44
}
x_plot_multi_product_diffusion__mutmut_orig.__name__ = 'x_plot_multi_product_diffusion'
