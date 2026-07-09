"""Fitters module for parameter estimation."""

from typing import Any

from .batched_fitter import BatchedFitter
from .bootstrap_fitter import BootstrapFitter
from .curve_fitter import CurveFitter
from .diagnostics_contract import (
    DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION,
    DiagnosticsArtifactPayload,
    DiagnosticsContract,
    DiagnosticsWarning,
    IntervalEstimates,
    UncertaintySummary,
    build_diagnostics_contract,
)
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
BayesianFitter: Any
try:
    from .bayesian_fitter import BayesianFitter as _BayesianFitter
except ImportError:
    BayesianFitter = _MissingOptionalDependency("BayesianFitter", "innovate[bayesian]")
else:
    BayesianFitter = _BayesianFitter

BlackJaxFitter: Any
try:
    from .blackjax_fitter import BlackJaxFitter as _BlackJaxFitter
except ImportError:
    BlackJaxFitter = _MissingOptionalDependency("BlackJaxFitter", "innovate[bayesian]")
else:
    BlackJaxFitter = _BlackJaxFitter

JaxFitter: Any
try:
    from .jax_fitter import JaxFitter as _JaxFitter
except ImportError:
    JaxFitter = _MissingOptionalDependency("JaxFitter", "innovate[jax]")
else:
    JaxFitter = _JaxFitter

__all__ = [
    "DIAGNOSTICS_ARTIFACT_SCHEMA_VERSION",
    "BatchedFitter",
    "BayesianFitter",
    "BlackJaxFitter",
    "BootstrapFitter",
    "CurveFitter",
    "DiagnosticsArtifactPayload",
    "DiagnosticsContract",
    "DiagnosticsWarning",
    "IntervalEstimates",
    "JaxFitter",
    "MoMFitter",
    "ScipyFitter",
    "UncertaintySummary",
    "build_diagnostics_contract",
]
