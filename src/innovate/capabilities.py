"""Stable capability metadata for canonical model families and backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Machine-readable summary of a stable model family's public contract."""

    key: str
    family: str
    import_path: str
    supports_covariates: bool
    supports_multivariate_output: bool = False
    stability: str = "stable"
    optional_dependencies: tuple[str, ...] = ()
    supported_backends: tuple[str, ...] = ("numpy", "jax")


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """Machine-readable summary of a runtime backend."""

    key: str
    description: str
    stability: str = "stable"
    optional_dependencies: tuple[str, ...] = ()
    available: bool = True


@dataclass(frozen=True, slots=True)
class FitterCapability:
    """Machine-readable summary of a fitter family."""

    key: str
    family: str
    import_path: str
    supported_backends: tuple[str, ...]
    stability: str = "stable"
    optional_dependencies: tuple[str, ...] = ()
    available: bool = True


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


_MODEL_REGISTRY = MappingProxyType(
    {
        "bass": ModelCapability(
            key="bass",
            family="diffusion",
            import_path="innovate.diffuse.BassModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
        ),
        "logistic": ModelCapability(
            key="logistic",
            family="diffusion",
            import_path="innovate.diffuse.LogisticModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
        ),
        "gompertz": ModelCapability(
            key="gompertz",
            family="diffusion",
            import_path="innovate.diffuse.GompertzModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
        ),
        "fisher_pry": ModelCapability(
            key="fisher_pry",
            family="substitution",
            import_path="innovate.substitute.FisherPryModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=False,
        ),
        "norton_bass": ModelCapability(
            key="norton_bass",
            family="substitution",
            import_path="innovate.substitute.NortonBassModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "composite": ModelCapability(
            key="composite",
            family="substitution",
            import_path="innovate.substitute.CompositeDiffusionModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=False,
            supports_multivariate_output=True,
        ),
        "multi_product": ModelCapability(
            key="multi_product",
            family="competition",
            import_path="innovate.compete.MultiProductDiffusionModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "lotka_volterra": ModelCapability(
            key="lotka_volterra",
            family="competition",
            import_path="innovate.compete.LotkaVolterraModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "complementary_goods": ModelCapability(
            key="complementary_goods",
            family="ecosystem",
            import_path="innovate.ecosystem.ComplementaryGoodsModel",
            supported_backends=("numpy", "jax"),
            supports_covariates=False,
            supports_multivariate_output=True,
        ),
    },
)


_BACKEND_REGISTRY = MappingProxyType(
    {
        "numpy": BackendCapability(
            key="numpy",
            description="Reference NumPy/SciPy execution backend.",
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "jax": BackendCapability(
            key="jax",
            description="Optional accelerator backend backed by JAX and XLA.",
            stability="experimental",
            optional_dependencies=("jax", "jaxlib", "diffrax"),
            available=_module_available("jax")
            and _module_available("jaxlib")
            and _module_available("diffrax"),
        ),
    },
)


_FITTER_REGISTRY = MappingProxyType(
    {
        "scipy": FitterCapability(
            key="scipy",
            family="optimization",
            import_path="innovate.fitters.ScipyFitter",
            supported_backends=("numpy",),
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "bootstrap": FitterCapability(
            key="bootstrap",
            family="uncertainty",
            import_path="innovate.fitters.BootstrapFitter",
            supported_backends=("numpy",),
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "mom": FitterCapability(
            key="mom",
            family="moment",
            import_path="innovate.fitters.MoMFitter",
            supported_backends=("numpy",),
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "curve": FitterCapability(
            key="curve",
            family="optimization",
            import_path="innovate.fitters.CurveFitter",
            supported_backends=("numpy",),
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "batched": FitterCapability(
            key="batched",
            family="batching",
            import_path="innovate.fitters.BatchedFitter",
            supported_backends=("numpy", "jax"),
            stability="stable",
            optional_dependencies=(),
            available=True,
        ),
        "jax": FitterCapability(
            key="jax",
            family="acceleration",
            import_path="innovate.fitters.JaxFitter",
            supported_backends=("jax",),
            stability="experimental",
            optional_dependencies=("jax", "jaxlib", "jaxopt"),
            available=_module_available("jax")
            and _module_available("jaxlib")
            and _module_available("jaxopt"),
        ),
        "bayesian": FitterCapability(
            key="bayesian",
            family="inference",
            import_path="innovate.fitters.BayesianFitter",
            supported_backends=("jax",),
            stability="experimental",
            optional_dependencies=("jax", "jaxlib", "blackjax", "arviz"),
            available=_module_available("jax")
            and _module_available("jaxlib")
            and _module_available("blackjax")
            and _module_available("arviz"),
        ),
        "blackjax": FitterCapability(
            key="blackjax",
            family="inference",
            import_path="innovate.fitters.BlackJaxFitter",
            supported_backends=("jax",),
            stability="experimental",
            optional_dependencies=("jax", "jaxlib", "blackjax", "arviz"),
            available=_module_available("jax")
            and _module_available("jaxlib")
            and _module_available("blackjax")
            and _module_available("arviz"),
        ),
    },
)


def get_model_registry() -> Mapping[str, ModelCapability]:
    """Return the immutable registry for stable model families."""
    return _MODEL_REGISTRY


def get_model_capability(key: str) -> ModelCapability:
    """Look up one model family by registry key."""
    try:
        return _MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(f"Unknown model capability {key!r}. Available models: {available}") from exc


def get_backend_registry() -> Mapping[str, BackendCapability]:
    """Return the immutable registry for runtime backends."""
    return _BACKEND_REGISTRY


def get_backend_capability(key: str) -> BackendCapability:
    """Look up one runtime backend by registry key."""
    try:
        return _BACKEND_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise KeyError(f"Unknown backend capability {key!r}. Available backends: {available}") from exc


def get_fitter_registry() -> Mapping[str, FitterCapability]:
    """Return the immutable registry for fitter families."""
    return _FITTER_REGISTRY


def get_fitter_capability(key: str) -> FitterCapability:
    """Look up one fitter family by registry key."""
    try:
        return _FITTER_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_FITTER_REGISTRY))
        raise KeyError(f"Unknown fitter capability {key!r}. Available fitters: {available}") from exc
