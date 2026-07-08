"""Probabilistic inference payloads and optional backend metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from innovate.fitters.diagnostics_contract import UncertaintySummary

PROBABILISTIC_SCHEMA_MAJOR_VERSION = 1
PROBABILISTIC_SCHEMA_MINOR_VERSION = 0
PROBABILISTIC_SCHEMA_VERSION = f"{PROBABILISTIC_SCHEMA_MAJOR_VERSION}.{PROBABILISTIC_SCHEMA_MINOR_VERSION}"
PROBABILISTIC_INSTALL_HINT = "innovate[bayesian]"


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _validate_schema_version(schema_version: str) -> str:
    if schema_version != PROBABILISTIC_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported probabilistic schema version: {schema_version}. "
            f"Supported version is {PROBABILISTIC_SCHEMA_VERSION}",
        )
    return schema_version


@dataclass(frozen=True, slots=True)
class ProbabilisticBackendStatus:
    """Machine-readable status for an optional probabilistic engine."""

    engine: str
    role: str
    backend: str = "jax"
    xla_eligible: bool = True
    optional_dependencies: tuple[str, ...] = ()
    available: bool = False
    install_hint: str = PROBABILISTIC_INSTALL_HINT
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the backend status to a stable dictionary."""
        return {
            "engine": self.engine,
            "role": self.role,
            "backend": self.backend,
            "xla_eligible": self.xla_eligible,
            "optional_dependencies": list(self.optional_dependencies),
            "available": self.available,
            "install_hint": self.install_hint,
            "note": self.note,
        }


class ProbabilisticBackendUnavailableError(ImportError):
    """Structured error raised when an optional probabilistic backend is absent."""

    def __init__(
        self,
        *,
        engine: str,
        missing_dependencies: tuple[str, ...],
        install_hint: str = PROBABILISTIC_INSTALL_HINT,
    ) -> None:
        self.engine = engine
        self.missing_dependencies = missing_dependencies
        self.install_hint = install_hint
        missing = ", ".join(missing_dependencies)
        super().__init__(f"{engine} is unavailable; install {install_hint}. Missing dependencies: {missing}")

    def to_dict(self) -> dict[str, object]:
        """Serialize the missing-backend error for kernel and binding payloads."""
        return {
            "code": "probabilistic_backend_unavailable",
            "engine": self.engine,
            "message": str(self),
            "missing_dependencies": list(self.missing_dependencies),
            "install_hint": self.install_hint,
        }


def list_probabilistic_backend_statuses() -> tuple[ProbabilisticBackendStatus, ...]:
    """Return optional probabilistic backend status without importing heavy extras."""
    backend_specs = (
        (
            "numpyro",
            "probabilistic_programming",
            ("jax", "jaxlib", "numpyro", "arviz"),
            "Preferred first-class modelling path for JAX/XLA probabilistic inference.",
        ),
        (
            "blackjax",
            "sampler",
            ("jax", "jaxlib", "blackjax", "arviz"),
            "Preferred low-level sampler path when innovate owns the log-density.",
        ),
        (
            "tensorflow_probability_jax",
            "distribution_bijector",
            ("jax", "jaxlib", "tensorflow_probability"),
            "Optional distribution and bijector coverage for JAX-backed workflows.",
        ),
    )
    statuses = []
    for engine, role, dependencies, note in backend_specs:
        statuses.append(
            ProbabilisticBackendStatus(
                engine=engine,
                role=role,
                optional_dependencies=dependencies,
                available=all(_module_available(dependency) for dependency in dependencies),
                note=note,
            ),
        )
    return tuple(statuses)


def require_probabilistic_backend(
    *,
    engine: str,
    optional_dependencies: tuple[str, ...],
    install_hint: str = PROBABILISTIC_INSTALL_HINT,
) -> None:
    """Raise a structured error when an optional probabilistic backend is missing."""
    missing = tuple(dependency for dependency in optional_dependencies if not _module_available(dependency))
    if missing:
        raise ProbabilisticBackendUnavailableError(
            engine=engine,
            missing_dependencies=missing,
            install_hint=install_hint,
        )


@dataclass(frozen=True, slots=True)
class PosteriorConfig:
    """Configuration for posterior samples generation."""

    engine: str = "blackjax"
    backend: str = "jax"
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PosteriorSamplesPayload:
    """Versioned posterior draws for schema-compatible probabilistic outputs."""

    model_key: str
    parameter_names: tuple[str, ...]
    draw_shape: tuple[int, int]
    samples: Mapping[str, tuple[float, ...]]
    engine: str = "blackjax"
    backend: str = "jax"
    provenance: str = "bayesian"
    schema_version: str = PROBABILISTIC_SCHEMA_VERSION
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        if not self.model_key:
            raise ValueError("Posterior payload model_key must be non-empty")
        if not self.parameter_names:
            raise ValueError("Posterior payload must include parameter names")
        if len(self.draw_shape) != 2 or self.draw_shape[0] <= 0 or self.draw_shape[1] <= 0:
            raise ValueError("Posterior payload draw_shape must be positive (chains, draws)")

        normalized_samples: dict[str, tuple[float, ...]] = {}
        expected_size = self.draw_shape[0] * self.draw_shape[1]
        for name in self.parameter_names:
            values = tuple(float(value) for value in self.samples[name])
            if len(values) != expected_size:
                raise ValueError("Posterior sample values must match draw_shape")
            normalized_samples[name] = values

        object.__setattr__(self, "samples", MappingProxyType(normalized_samples))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_samples(
        cls,
        *,
        model_key: str,
        samples: Mapping[str, Any],
        config: PosteriorConfig | None = None,
    ) -> PosteriorSamplesPayload:
        """Build a posterior payload from parameter draws shaped as chains x draws."""
        if not samples:
            raise ValueError("Posterior payload requires at least one parameter sample")

        if config is None:
            config = PosteriorConfig()

        arrays = {name: np.asarray(values, dtype=float) for name, values in samples.items()}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1 or any(len(shape) != 2 for shape in shapes):
            raise ValueError("Posterior parameters must share the same 2D draw shape")

        draw_shape = next(iter(shapes))
        flattened = {name: tuple(array.reshape(-1).tolist()) for name, array in arrays.items()}
        return cls(
            model_key=model_key,
            parameter_names=tuple(arrays),
            draw_shape=(int(draw_shape[0]), int(draw_shape[1])),
            samples=flattened,
            engine=config.engine,
            backend=config.backend,
            seed=config.seed,
            metadata=config.metadata,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PosteriorSamplesPayload:
        """Deserialize a posterior payload from a dictionary."""
        return cls(
            schema_version=str(payload.get("schema_version", PROBABILISTIC_SCHEMA_VERSION)),
            model_key=str(payload["model_key"]),
            parameter_names=tuple(str(name) for name in payload["parameter_names"]),
            draw_shape=tuple(int(dim) for dim in payload["draw_shape"]),  # type: ignore[arg-type]
            samples={
                str(name): tuple(float(value) for value in values) for name, values in dict(payload["samples"]).items()
            },
            engine=str(payload.get("engine", "blackjax")),
            backend=str(payload.get("backend", "jax")),
            provenance=str(payload.get("provenance", "bayesian")),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize posterior samples to a JSON-compatible payload."""
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key,
            "parameter_names": list(self.parameter_names),
            "draw_shape": list(self.draw_shape),
            "samples": {name: list(values) for name, values in self.samples.items()},
            "engine": self.engine,
            "backend": self.backend,
            "provenance": self.provenance,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    def sample_array(self, parameter_name: str) -> np.ndarray:
        """Return one parameter's samples with shape ``chains x draws``."""
        return np.asarray(self.samples[parameter_name], dtype=float).reshape(self.draw_shape)

    def to_uncertainty_summary(self, *, level: float = 0.95) -> UncertaintySummary:
        """Convert the posterior payload into the shared uncertainty summary."""
        alpha = 1.0 - level
        lower_percentile = 100.0 * alpha / 2.0
        upper_percentile = 100.0 * (1.0 - alpha / 2.0)
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        median: dict[str, float] = {}
        summary_samples: dict[str, np.ndarray] = {}

        for parameter_name in self.parameter_names:
            draws = self.sample_array(parameter_name).reshape(-1)
            lower[parameter_name] = float(np.percentile(draws, lower_percentile))
            upper[parameter_name] = float(np.percentile(draws, upper_percentile))
            median[parameter_name] = float(np.median(draws))
            summary_samples[parameter_name] = draws

        return UncertaintySummary.posterior_summary(
            lower=lower,
            upper=upper,
            median=median,
            level=level,
            samples=summary_samples,
            note=(
                f"{self.draw_shape[0]} chains x {self.draw_shape[1]} draws; "
                f"engine={self.engine}; backend={self.backend}"
            ),
        )


__all__ = [
    "PROBABILISTIC_INSTALL_HINT",
    "PROBABILISTIC_SCHEMA_MAJOR_VERSION",
    "PROBABILISTIC_SCHEMA_MINOR_VERSION",
    "PROBABILISTIC_SCHEMA_VERSION",
    "PosteriorConfig",
    "PosteriorSamplesPayload",
    "ProbabilisticBackendStatus",
    "ProbabilisticBackendUnavailable",
    "ProbabilisticBackendUnavailableError",
    "list_probabilistic_backend_statuses",
    "require_probabilistic_backend",
]

ProbabilisticBackendUnavailable = ProbabilisticBackendUnavailableError
