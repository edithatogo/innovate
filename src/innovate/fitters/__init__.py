"""Fitters module for parameter estimation."""

from .batched_fitter import BatchedFitter
from .bootstrap_fitter import BootstrapFitter
from .curve_fitter import CurveFitter
from .mom_fitter import MoMFitter
from .scipy_fitter import ScipyFitter


class _MissingOptionalDependency:
    """Proxy object that raises a clear import error when used."""

    def __init__(self, feature: str, install_hint: str):
        self._feature = feature
        self._install_hint = install_hint

    def _raise(self) -> None:
        raise ImportError(
            f"{self._feature} is not available. Install {self._install_hint} to enable it.",
        )

    def __call__(self, *args, **kwargs):  # pragma: no cover - trivial proxy
        self._raise()

    def __getattr__(self, name: str):  # pragma: no cover - trivial proxy
        self._raise()

    def __repr__(self) -> str:  # pragma: no cover - trivial proxy
        return f"<missing optional dependency proxy for {self._feature}>"


# Optional fitters — only imported when their dependencies are available
try:
    from .bayesian_fitter import BayesianFitter
except ImportError:
    BayesianFitter = _MissingOptionalDependency("BayesianFitter", "innovate[bayesian]")

try:
    from .blackjax_fitter import BlackJaxFitter
except ImportError:
    BlackJaxFitter = _MissingOptionalDependency("BlackJaxFitter", "innovate[bayesian]")

try:
    from .jax_fitter import JaxFitter
except ImportError:
    JaxFitter = _MissingOptionalDependency("JaxFitter", "innovate[jax]")

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
