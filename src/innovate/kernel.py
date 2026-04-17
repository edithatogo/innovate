"""Functional kernel contract for language-neutral model execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Any, TypeAlias

from .capabilities import ModelCapability, get_model_registry

KernelJSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | dict[str, "KernelJSONValue"]
    | list["KernelJSONValue"]
)

KERNEL_SCHEMA_VERSION = "1.0"
KERNEL_OPERATIONS = (
    "discover_models",
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
)


class KernelOperation(str, Enum):
    """Canonical kernel operations."""

    DISCOVER_MODELS = "discover_models"
    FIT_MODEL = "fit_model"
    PREDICT_MODEL = "predict_model"
    SIMULATE_MODEL = "simulate_model"
    SUMMARIZE_MODEL = "summarize_model"
    DIAGNOSE_MODEL = "diagnose_model"


class KernelErrorCode(str, Enum):
    """Stable error codes for kernel responses."""

    INVALID_REQUEST = "invalid_request"
    INVALID_SCHEMA_VERSION = "invalid_schema_version"
    UNAVAILABLE_MODEL = "unavailable_model"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INVALID_PAYLOAD = "invalid_payload"
    INTERNAL_ERROR = "internal_error"


def _validate_schema_version(schema_version: str) -> str:
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("Kernel schema version must be a non-empty string")

    parts = schema_version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("Kernel schema version must use major.minor notation")
    if int(parts[0]) != 1:
        raise ValueError(f"Unsupported kernel schema version: {schema_version}")
    return schema_version


def _validate_operation(operation: str) -> str:
    if operation not in KERNEL_OPERATIONS:
        raise ValueError(f"Unknown kernel operation: {operation}")
    return operation


def _copy_mapping(values: Mapping[str, KernelJSONValue] | None) -> dict[str, KernelJSONValue]:
    return dict(values or {})


def _as_dict_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _as_dict_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_dict_value(item) for item in value]
    if isinstance(value, tuple):
        return [_as_dict_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class KernelError:
    """Stable error payload returned by kernel operations."""

    code: str
    message: str
    operation: str | None = None
    details: dict[str, KernelJSONValue] = field(default_factory=dict)
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("Kernel error code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Kernel error message must be a non-empty string")
        if self.operation is not None:
            _validate_operation(self.operation)
        object.__setattr__(self, "details", _copy_mapping(self.details))

    def to_dict(self) -> dict[str, object]:
        """Serialize the error payload to a JSON-friendly dictionary."""
        payload = {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "details": _as_dict_value(self.details),
            "retryable": self.retryable,
        }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelError:
        """Construct an error payload from a JSON-friendly dictionary."""
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            operation=None if data.get("operation") is None else str(data["operation"]),
            details=_copy_mapping(data.get("details") if isinstance(data.get("details"), Mapping) else {}),
            retryable=bool(data.get("retryable", False)),
        )


@dataclass(frozen=True, slots=True)
class KernelArrayPayload:
    """Portable numeric array payload for kernel requests and responses."""

    shape: tuple[int, ...]
    dtype: str
    values: tuple[float, ...]
    metadata: dict[str, KernelJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("Kernel arrays must declare a shape")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("Kernel array dtype must be a non-empty string")
        if prod(self.shape) != len(self.values):
            raise ValueError("Kernel array shape must match the number of values")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        shape: Sequence[int],
        dtype: str,
        metadata: Mapping[str, KernelJSONValue] | None = None,
    ) -> KernelArrayPayload:
        """Build a kernel array payload from a flat value sequence."""
        return cls(
            shape=tuple(int(dimension) for dimension in shape),
            dtype=dtype,
            values=tuple(float(value) for value in values),
            metadata=_copy_mapping(metadata),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the array payload to a JSON-friendly dictionary."""
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "values": list(self.values),
            "metadata": _as_dict_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelArrayPayload:
        """Construct an array payload from a JSON-friendly dictionary."""
        return cls.from_values(
            values=data["values"],
            shape=data["shape"],
            dtype=str(data["dtype"]),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class KernelTablePayload:
    """Arrow-friendly tabular payload for kernel responses."""

    columns: tuple[str, ...]
    rows: tuple[tuple[KernelJSONValue, ...], ...]
    metadata: dict[str, KernelJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("Kernel tables must declare at least one column")
        if any(not isinstance(column, str) or not column.strip() for column in self.columns):
            raise ValueError("Kernel table columns must be non-empty strings")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Kernel table rows must match the number of columns")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_rows(
        cls,
        columns: Sequence[str],
        rows: Sequence[Sequence[KernelJSONValue]],
        metadata: Mapping[str, KernelJSONValue] | None = None,
    ) -> KernelTablePayload:
        """Build a tabular payload from row-oriented data."""
        return cls(
            columns=tuple(str(column) for column in columns),
            rows=tuple(tuple(row) for row in rows),
            metadata=_copy_mapping(metadata),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the table payload to a JSON-friendly dictionary."""
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "metadata": _as_dict_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelTablePayload:
        """Construct a table payload from a JSON-friendly dictionary."""
        return cls.from_rows(
            columns=data["columns"],
            rows=data["rows"],
            metadata=data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class KernelRequest:
    """Versioned request envelope for kernel operations."""

    operation: str
    model_key: str | None
    payload: dict[str, KernelJSONValue]
    schema_version: str = KERNEL_SCHEMA_VERSION
    metadata: dict[str, KernelJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "operation", _validate_operation(self.operation))
        if self.operation != KernelOperation.DISCOVER_MODELS.value and not self.model_key:
            raise ValueError(f"Kernel operation '{self.operation}' requires a model_key")
        object.__setattr__(self, "payload", _copy_mapping(self.payload))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Serialize the request envelope to a JSON-friendly dictionary."""
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "model_key": self.model_key,
            "payload": _as_dict_value(self.payload),
            "metadata": _as_dict_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelRequest:
        """Construct a request envelope from a JSON-friendly dictionary."""
        return cls(
            schema_version=str(data.get("schema_version", KERNEL_SCHEMA_VERSION)),
            operation=str(data["operation"]),
            model_key=None if data.get("model_key") is None else str(data["model_key"]),
            payload=_copy_mapping(data.get("payload") if isinstance(data.get("payload"), Mapping) else {}),
            metadata=_copy_mapping(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}),
        )


@dataclass(frozen=True, slots=True)
class KernelResponse:
    """Versioned response envelope for kernel operations."""

    operation: str
    result: dict[str, KernelJSONValue] | KernelArrayPayload | KernelTablePayload | None = None
    error: KernelError | None = None
    schema_version: str = KERNEL_SCHEMA_VERSION
    metadata: dict[str, KernelJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "operation", _validate_operation(self.operation))
        if self.error is not None and not isinstance(self.error, KernelError):
            raise TypeError("Kernel response errors must be KernelError instances")
        if self.error is None and self.result is None:
            raise ValueError("Kernel responses require either a result or an error")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Serialize the response envelope to a JSON-friendly dictionary."""
        if isinstance(self.result, (KernelArrayPayload, KernelTablePayload)):
            result: Any = self.result.to_dict()
        else:
            result = _as_dict_value(self.result)
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "result": result,
            "error": None if self.error is None else self.error.to_dict(),
            "metadata": _as_dict_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelResponse:
        """Construct a response envelope from a JSON-friendly dictionary."""
        result = data.get("result")
        if isinstance(result, Mapping) and {"shape", "dtype", "values"}.issubset(result):
            result = KernelArrayPayload.from_dict(result)
        elif isinstance(result, Mapping) and {"columns", "rows"}.issubset(result):
            result = KernelTablePayload.from_dict(result)

        error = data.get("error")
        if isinstance(error, Mapping):
            error = KernelError.from_dict(error)

        return cls(
            schema_version=str(data.get("schema_version", KERNEL_SCHEMA_VERSION)),
            operation=str(data["operation"]),
            result=result,
            error=error,
            metadata=_copy_mapping(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}),
        )


@dataclass(frozen=True, slots=True)
class KernelDiscoveryRecord:
    """Serializable metadata describing a discoverable model family."""

    key: str
    family: str
    import_path: str
    stability: str
    supports_covariates: bool
    supports_multivariate_output: bool
    supported_backends: tuple[str, ...]
    optional_dependencies: tuple[str, ...] = ()
    supports_simulation: bool = False
    supports_summarize: bool = False

    @classmethod
    def from_capability(cls, capability: ModelCapability) -> KernelDiscoveryRecord:
        """Build a discovery record from a canonical model capability."""
        return cls(
            key=capability.key,
            family=capability.family,
            import_path=capability.import_path,
            stability=capability.stability_tier.value,
            supports_covariates=capability.supports_covariates,
            supports_multivariate_output=capability.supports_multivariate_output,
            supported_backends=tuple(capability.supported_backends),
            optional_dependencies=tuple(capability.optional_dependencies),
            supports_simulation=capability.supports_simulation,
            supports_summarize=capability.supports_summarize,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the discovery record to a JSON-friendly dictionary."""
        return {
            "key": self.key,
            "family": self.family,
            "import_path": self.import_path,
            "stability": self.stability,
            "supports_covariates": self.supports_covariates,
            "supports_multivariate_output": self.supports_multivariate_output,
            "supported_backends": list(self.supported_backends),
            "optional_dependencies": list(self.optional_dependencies),
            "supports_simulation": self.supports_simulation,
            "supports_summarize": self.supports_summarize,
        }


@dataclass(frozen=True, slots=True)
class KernelDiscoveryResponse:
    """Versioned response envelope for model discovery."""

    models: tuple[KernelDiscoveryRecord, ...]
    schema_version: str = KERNEL_SCHEMA_VERSION
    metadata: dict[str, KernelJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validate_schema_version(self.schema_version))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Serialize the discovery response to a JSON-friendly dictionary."""
        return {
            "schema_version": self.schema_version,
            "models": [record.to_dict() for record in self.models],
            "metadata": _as_dict_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelDiscoveryResponse:
        """Construct a discovery response from a JSON-friendly dictionary."""
        models = tuple(
            KernelDiscoveryRecord(
                key=str(record["key"]),
                family=str(record["family"]),
                import_path=str(record["import_path"]),
                stability=str(record["stability"]),
                supports_covariates=bool(record["supports_covariates"]),
                supports_multivariate_output=bool(record["supports_multivariate_output"]),
                supported_backends=tuple(str(value) for value in record["supported_backends"]),
                optional_dependencies=tuple(str(value) for value in record.get("optional_dependencies", ())),
                supports_simulation=bool(record.get("supports_simulation", False)),
                supports_summarize=bool(record.get("supports_summarize", False)),
            )
            for record in data.get("models", [])
        )
        return cls(
            schema_version=str(data.get("schema_version", KERNEL_SCHEMA_VERSION)),
            models=models,
            metadata=_copy_mapping(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}),
        )


def list_kernel_operations() -> tuple[str, ...]:
    """Return the canonical kernel operation names."""
    return KERNEL_OPERATIONS


def discover_models() -> KernelDiscoveryResponse:
    """Return machine-readable discovery metadata for canonical model families."""
    records = tuple(
        KernelDiscoveryRecord.from_capability(capability)
        for capability in get_model_registry().values()
    )
    return KernelDiscoveryResponse(models=records)


def _unsupported_kernel_operation(operation: KernelOperation) -> None:
    raise NotImplementedError(
        f"Kernel operation '{operation.value}' is not implemented yet",
    )


def fit_model(request: KernelRequest) -> KernelResponse:
    """Placeholder for the fit operation adapter."""
    _unsupported_kernel_operation(KernelOperation.FIT_MODEL)


def predict_model(request: KernelRequest) -> KernelResponse:
    """Placeholder for the predict operation adapter."""
    _unsupported_kernel_operation(KernelOperation.PREDICT_MODEL)


def simulate_model(request: KernelRequest) -> KernelResponse:
    """Placeholder for the simulate operation adapter."""
    _unsupported_kernel_operation(KernelOperation.SIMULATE_MODEL)


def summarize_model(request: KernelRequest) -> KernelResponse:
    """Placeholder for the summarize operation adapter."""
    _unsupported_kernel_operation(KernelOperation.SUMMARIZE_MODEL)


def diagnose_model(request: KernelRequest) -> KernelResponse:
    """Placeholder for the diagnostics operation adapter."""
    _unsupported_kernel_operation(KernelOperation.DIAGNOSE_MODEL)


__all__ = [
    "KERNEL_OPERATIONS",
    "KERNEL_SCHEMA_VERSION",
    "KernelArrayPayload",
    "KernelDiscoveryRecord",
    "KernelDiscoveryResponse",
    "KernelError",
    "KernelErrorCode",
    "KernelJSONValue",
    "KernelOperation",
    "KernelRequest",
    "KernelResponse",
    "KernelTablePayload",
    "diagnose_model",
    "discover_models",
    "fit_model",
    "list_kernel_operations",
    "predict_model",
    "simulate_model",
    "summarize_model",
]
