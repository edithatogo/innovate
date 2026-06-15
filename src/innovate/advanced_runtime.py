"""Advanced modeling capability and runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from types import MappingProxyType
from typing import Any

_VALID_STABILITY = frozenset({"stable", "experimental"})
ADVANCED_RESULT_SCHEMA_VERSION = "advanced_result.v1"


def _validate_stability(stability: str) -> str:
    if stability not in _VALID_STABILITY:
        raise ValueError(f"Unknown advanced API stability label: {stability!r}")
    return stability


def _json_dict(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): value for key, value in dict(values or {}).items()}


@dataclass(frozen=True, slots=True)
class AdvancedCapability:
    """Machine-readable contract metadata for an advanced workflow."""

    key: str
    family: str
    stability: str
    result_schema: str = ADVANCED_RESULT_SCHEMA_VERSION
    optional_dependencies: tuple[str, ...] = ()
    supported_backends: tuple[str, ...] = ("numpy",)
    supports_incremental: bool = False
    supports_uncertainty: bool = False

    def __post_init__(self) -> None:
        _validate_stability(self.stability)
        if not self.key.strip():
            raise ValueError("Advanced capability key must be non-empty")
        if not self.family.strip():
            raise ValueError("Advanced capability family must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize the capability metadata to a JSON-friendly payload."""
        return {
            "key": self.key,
            "family": self.family,
            "stability": self.stability,
            "result_schema": self.result_schema,
            "optional_dependencies": list(self.optional_dependencies),
            "supported_backends": list(self.supported_backends),
            "supports_incremental": self.supports_incremental,
            "supports_uncertainty": self.supports_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class AdvancedResult:
    """Stable JSON-friendly result payload for advanced workflows."""

    workflow: str
    stability: str
    backend: str
    time: Sequence[float] = ()
    mean: Sequence[float] = ()
    lower: Sequence[float] | None = None
    upper: Sequence[float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ADVANCED_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_stability(self.stability)
        if not self.workflow.strip():
            raise ValueError("Advanced result workflow must be non-empty")
        if not self.backend.strip():
            raise ValueError("Advanced result backend must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize the result to a stable JSON-friendly payload."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "capability": {
                "workflow": self.workflow,
                "stability": self.stability,
                "backend": self.backend,
            },
            "time": [float(value) for value in self.time],
            "mean": [float(value) for value in self.mean],
            "metadata": _json_dict(self.metadata),
            "diagnostics": _json_dict(self.diagnostics),
        }
        if self.lower is not None:
            payload["lower"] = [float(value) for value in self.lower]
        if self.upper is not None:
            payload["upper"] = [float(value) for value in self.upper]
        return payload


@dataclass(frozen=True, slots=True)
class AdvancedRuntimePolicy:
    """Backend preference policy for advanced workflow execution."""

    preferred: tuple[str, ...] = ("numpy",)
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if not self.preferred:
            raise ValueError("Advanced runtime policy requires at least one preferred backend")


@dataclass(frozen=True, slots=True)
class AdvancedBackendSelection:
    """Resolved backend selection and fallback metadata."""

    backend: str
    requested_backend: str
    fallback_used: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize backend selection metadata."""
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
        }


_ADVANCED_CAPABILITIES = MappingProxyType(
    {
        "regime_ensemble": AdvancedCapability(
            key="regime_ensemble",
            family="ensemble",
            stability="experimental",
            supported_backends=("numpy", "jax", "rust"),
        ),
        "policy_scenario": AdvancedCapability(
            key="policy_scenario",
            family="policy",
            stability="stable",
            supported_backends=("numpy", "jax", "rust"),
        ),
        "streaming_update": AdvancedCapability(
            key="streaming_update",
            family="streaming",
            stability="experimental",
            supported_backends=("numpy", "rust"),
            supports_incremental=True,
        ),
        "uncertainty_calibration": AdvancedCapability(
            key="uncertainty_calibration",
            family="uncertainty",
            stability="stable",
            supported_backends=("numpy", "jax", "rust"),
            supports_uncertainty=True,
        ),
    },
)


def list_advanced_capabilities() -> tuple[AdvancedCapability, ...]:
    """Return advanced workflow capability metadata in deterministic order."""
    return tuple(_ADVANCED_CAPABILITIES[key] for key in sorted(_ADVANCED_CAPABILITIES))


def get_advanced_capability(key: str) -> AdvancedCapability:
    """Return one advanced workflow capability by key."""
    try:
        return _ADVANCED_CAPABILITIES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown advanced workflow capability: {key}") from exc


def detect_advanced_backends() -> dict[str, bool]:
    """Detect optional backend availability for advanced workflows."""
    return {
        "numpy": True,
        "jax": find_spec("jax") is not None and find_spec("jaxlib") is not None,
        "rust": find_spec("innovate_rust") is not None,
    }


def select_advanced_backend(
    workflow: str,
    *,
    policy: AdvancedRuntimePolicy | None = None,
    available_backends: Mapping[str, bool] | None = None,
) -> AdvancedBackendSelection:
    """Resolve a backend for an advanced workflow with explicit fallback metadata."""
    capability = get_advanced_capability(workflow)
    runtime_policy = policy or AdvancedRuntimePolicy()
    availability = dict(detect_advanced_backends() if available_backends is None else available_backends)
    requested = runtime_policy.preferred[0]

    for backend in runtime_policy.preferred:
        if backend not in capability.supported_backends:
            continue
        if availability.get(backend, False):
            return AdvancedBackendSelection(
                backend=backend,
                requested_backend=requested,
                fallback_used=backend != requested,
                reason="preferred_backend_available" if backend == requested else "preferred_backend_unavailable",
            )

    if not runtime_policy.allow_fallback:
        raise RuntimeError(f"Requested backend {requested!r} is unavailable for advanced workflow {workflow!r}")

    for backend in capability.supported_backends:
        if availability.get(backend, False):
            return AdvancedBackendSelection(
                backend=backend,
                requested_backend=requested,
                fallback_used=backend != requested,
                reason="preferred_backend_unavailable",
            )

    raise RuntimeError(f"No available backend for advanced workflow {workflow!r}")
