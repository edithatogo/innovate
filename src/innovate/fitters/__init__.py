"""Fitters module for parameter estimation."""

from .batched_fitter import BatchedFitter
from .bootstrap_fitter import BootstrapFitter
from .curve_fitter import CurveFitter
from .mom_fitter import MoMFitter
from .scipy_fitter import ScipyFitter

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
    "BatchedFitter",
    "BayesianFitter",
    "BlackJaxFitter",
    "BootstrapFitter",
    "CurveFitter",
    "JaxFitter",
    "MoMFitter",
    "ScipyFitter",
]
