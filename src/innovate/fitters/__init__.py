"""Fitters module for parameter estimation."""

from .scipy_fitter import ScipyFitter
from .bootstrap_fitter import BootstrapFitter
from .mom_fitter import MoMFitter
from .batched_fitter import BatchedFitter
from .curve_fitter import CurveFitter

# Optional fitters — only imported when their dependencies are available
try:
    from .bayesian_fitter import BayesianFitter
except ImportError:
    BayesianFitter = None  # type: ignore[misc,assignment]

try:
    from .blackjax_fitter import BlackJaxFitter
except ImportError:
    BlackJaxFitter = None  # type: ignore[misc,assignment]

try:
    from .jax_fitter import JaxFitter
except ImportError:
    JaxFitter = None  # type: ignore[misc,assignment]

__all__ = [
    "ScipyFitter",
    "BayesianFitter",
    "BlackJaxFitter",
    "BootstrapFitter",
    "MoMFitter",
    "JaxFitter",
    "BatchedFitter",
    "CurveFitter",
]
