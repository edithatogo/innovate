"""Stable capability metadata for canonical model families."""

from __future__ import annotations

from dataclasses import dataclass
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


_MODEL_REGISTRY = MappingProxyType(
    {
        "bass": ModelCapability(
            key="bass",
            family="diffusion",
            import_path="innovate.diffuse.BassModel",
            supports_covariates=True,
        ),
        "logistic": ModelCapability(
            key="logistic",
            family="diffusion",
            import_path="innovate.diffuse.LogisticModel",
            supports_covariates=True,
        ),
        "gompertz": ModelCapability(
            key="gompertz",
            family="diffusion",
            import_path="innovate.diffuse.GompertzModel",
            supports_covariates=True,
        ),
        "fisher_pry": ModelCapability(
            key="fisher_pry",
            family="substitution",
            import_path="innovate.substitute.FisherPryModel",
            supports_covariates=False,
        ),
        "norton_bass": ModelCapability(
            key="norton_bass",
            family="substitution",
            import_path="innovate.substitute.NortonBassModel",
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "composite": ModelCapability(
            key="composite",
            family="substitution",
            import_path="innovate.substitute.CompositeDiffusionModel",
            supports_covariates=False,
            supports_multivariate_output=True,
        ),
        "multi_product": ModelCapability(
            key="multi_product",
            family="competition",
            import_path="innovate.compete.MultiProductDiffusionModel",
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "lotka_volterra": ModelCapability(
            key="lotka_volterra",
            family="competition",
            import_path="innovate.compete.LotkaVolterraModel",
            supports_covariates=True,
            supports_multivariate_output=True,
        ),
        "complementary_goods": ModelCapability(
            key="complementary_goods",
            family="ecosystem",
            import_path="innovate.ecosystem.ComplementaryGoodsModel",
            supports_covariates=False,
            supports_multivariate_output=True,
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
