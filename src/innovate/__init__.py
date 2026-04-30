"""Canonical public API for the :mod:`innovate` package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

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
from .plugins import ExtensionManifest, get_registered_extensions, register_extension
from .stability import StabilityTier, describe_stability_tier, normalize_stability_tier

try:
    __version__ = version("innovate")
except PackageNotFoundError:  # pragma: no cover - local source tree without install metadata
    __version__ = "0.0.0"

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "backend": ("innovate.backend", None),
    "backends": ("innovate.backends", None),
    "benchmarks": ("innovate.benchmarks", None),
    "arrow_interchange": ("innovate.arrow_interchange", None),
    "compete": ("innovate.compete", None),
    "diffuse": ("innovate.diffuse", None),
    "ecosystem": ("innovate.ecosystem", None),
    "fitters": ("innovate.fitters", None),
    "kernel": ("innovate.kernel", None),
    "models": ("innovate.models", None),
    "probabilistic": ("innovate.probabilistic", None),
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
    "AdvancedDiffusionModel": ("innovate.models", "AdvancedDiffusionModel"),
    "AdvancedModelSummary": ("innovate.models", "AdvancedModelSummary"),
    "BenchmarkCase": ("innovate.benchmarks", "BenchmarkCase"),
    "BenchmarkJob": ("innovate.benchmarks", "BenchmarkJob"),
    "BenchmarkFamily": ("innovate.benchmarks", "BenchmarkFamily"),
    "BenchmarkRun": ("innovate.benchmarks", "BenchmarkRun"),
    "BenchmarkRunner": ("innovate.benchmarks", "BenchmarkRunner"),
    "BenchmarkSuiteResult": ("innovate.benchmarks", "BenchmarkSuiteResult"),
    "ModelCard": ("innovate.benchmarks", "ModelCard"),
    "HierarchicalModel": ("innovate.models", "HierarchicalModel"),
    "NetworkDiffusionInputs": ("innovate.models", "NetworkDiffusionInputs"),
    "NetworkDiffusionModel": ("innovate.models", "NetworkDiffusionModel"),
    "LatentProcessDiffusionModel": ("innovate.models", "LatentProcessDiffusionModel"),
    "MixtureModel": ("innovate.models", "MixtureModel"),
    "PolicyHazardDiffusionModel": ("innovate.models", "PolicyHazardDiffusionModel"),
    "PolicyTimingInputs": ("innovate.models", "PolicyTimingInputs"),
    "RegimeSwitchingDiffusionModel": ("innovate.models", "RegimeSwitchingDiffusionModel"),
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
    "get_benchmark_case": ("innovate.benchmarks", "get_benchmark_case"),
    "list_benchmark_jobs": ("innovate.benchmarks", "list_benchmark_jobs"),
    "get_fitter_capability": ("innovate.capabilities", "get_fitter_capability"),
    "get_fitter_registry": ("innovate.capabilities", "get_fitter_registry"),
    "get_model_card": ("innovate.benchmarks", "get_model_card"),
    "list_benchmark_cases": ("innovate.benchmarks", "list_benchmark_cases"),
    "list_model_cards": ("innovate.benchmarks", "list_model_cards"),
    "run_stable_benchmark_suite": ("innovate.benchmarks", "run_stable_benchmark_suite"),
}

__all__ = [
    "AdvancedDiffusionModel",
    "AdvancedModelSummary",
    "BackendCapability",
    "BassModel",
    "BatchedFitter",
    "BayesianFitter",
    "BenchmarkCase",
    "BenchmarkFamily",
    "BenchmarkJob",
    "BenchmarkRun",
    "BenchmarkRunner",
    "BenchmarkSuiteResult",
    "BlackJaxFitter",
    "BootstrapFitter",
    "ComplementaryGoodsModel",
    "CompositeDiffusionModel",
    "CurveFitter",
    "DiffusionModel",
    "ExtensionManifest",
    "FisherPryModel",
    "FitterCapability",
    "GompertzModel",
    "HierarchicalModel",
    "JaxFitter",
    "LatentProcessDiffusionModel",
    "LogisticModel",
    "LotkaVolterraModel",
    "MixtureModel",
    "MoMFitter",
    "ModelCapability",
    "ModelCard",
    "MultiProductDiffusionModel",
    "NetworkDiffusionInputs",
    "NetworkDiffusionModel",
    "NortonBassModel",
    "PolicyHazardDiffusionModel",
    "PolicyTimingInputs",
    "RegimeSwitchingDiffusionModel",
    "ScipyFitter",
    "StabilityTier",
    "__version__",
    "arrow_interchange",
    "backend",
    "backends",
    "benchmarks",
    "compete",
    "describe_stability_tier",
    "diffuse",
    "ecosystem",
    "fitters",
    "get_backend_capability",
    "get_backend_registry",
    "get_benchmark_case",
    "get_fitter_capability",
    "get_fitter_registry",
    "get_model_capability",
    "get_model_card",
    "get_model_registry",
    "get_registered_extensions",
    "kernel",
    "list_benchmark_cases",
    "list_benchmark_jobs",
    "list_model_cards",
    "models",
    "normalize_stability_tier",
    "probabilistic",
    "register_extension",
    "run_stable_benchmark_suite",
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
