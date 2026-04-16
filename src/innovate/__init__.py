"""Canonical public API for the :mod:`innovate` package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import backend
from .base import DiffusionModel
from .capabilities import (
    BackendCapability,
    FitterCapability,
    ModelCapability,
    get_backend_capability,
    get_backend_registry,
    get_fitter_capability,
    get_fitter_registry,
    get_model_capability,
    get_model_registry,
)

try:
    __version__ = version("innovate")
except PackageNotFoundError:  # pragma: no cover - local source tree without install metadata
    __version__ = "0.0.0"

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "backends": ("innovate.backends", None),
    "compete": ("innovate.compete", None),
    "diffuse": ("innovate.diffuse", None),
    "ecosystem": ("innovate.ecosystem", None),
    "fitters": ("innovate.fitters", None),
    "substitute": ("innovate.substitute", None),
    "BassModel": ("innovate.diffuse", "BassModel"),
    "GompertzModel": ("innovate.diffuse", "GompertzModel"),
    "LogisticModel": ("innovate.diffuse", "LogisticModel"),
    "CompositeDiffusionModel": ("innovate.substitute", "CompositeDiffusionModel"),
    "FisherPryModel": ("innovate.substitute", "FisherPryModel"),
    "NortonBassModel": ("innovate.substitute", "NortonBassModel"),
    "LotkaVolterraModel": ("innovate.compete", "LotkaVolterraModel"),
    "MultiProductDiffusionModel": ("innovate.compete", "MultiProductDiffusionModel"),
    "ComplementaryGoodsModel": ("innovate.ecosystem", "ComplementaryGoodsModel"),
    "BatchedFitter": ("innovate.fitters", "BatchedFitter"),
    "BayesianFitter": ("innovate.fitters", "BayesianFitter"),
    "BlackJaxFitter": ("innovate.fitters", "BlackJaxFitter"),
    "BootstrapFitter": ("innovate.fitters", "BootstrapFitter"),
    "CurveFitter": ("innovate.fitters", "CurveFitter"),
    "JaxFitter": ("innovate.fitters", "JaxFitter"),
    "MoMFitter": ("innovate.fitters", "MoMFitter"),
    "ScipyFitter": ("innovate.fitters", "ScipyFitter"),
    "get_backend_capability": ("innovate.capabilities", "get_backend_capability"),
    "get_backend_registry": ("innovate.capabilities", "get_backend_registry"),
    "get_fitter_capability": ("innovate.capabilities", "get_fitter_capability"),
    "get_fitter_registry": ("innovate.capabilities", "get_fitter_registry"),
}

__all__ = [
    "BackendCapability",
    "BassModel",
    "BatchedFitter",
    "BayesianFitter",
    "BlackJaxFitter",
    "BootstrapFitter",
    "ComplementaryGoodsModel",
    "CompositeDiffusionModel",
    "CurveFitter",
    "DiffusionModel",
    "FisherPryModel",
    "FitterCapability",
    "GompertzModel",
    "JaxFitter",
    "LogisticModel",
    "LotkaVolterraModel",
    "MoMFitter",
    "ModelCapability",
    "MultiProductDiffusionModel",
    "NortonBassModel",
    "ScipyFitter",
    "__version__",
    "backend",
    "backends",
    "compete",
    "diffuse",
    "ecosystem",
    "fitters",
    "get_backend_capability",
    "get_backend_registry",
    "get_fitter_capability",
    "get_fitter_registry",
    "get_model_capability",
    "get_model_registry",
    "substitute",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the documented public API surface."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
