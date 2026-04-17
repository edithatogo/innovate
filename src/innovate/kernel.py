"""Functional kernel contract for language-neutral model execution."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from math import prod
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from .capabilities import ModelCapability, get_model_registry
from .fitters import ScipyFitter
from .fitters.diagnostics_contract import DiagnosticsContract, DiagnosticsWarning, UncertaintySummary
from .fitters.residual_analysis import analyze_residuals

KernelJSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | dict[str, "KernelJSONValue"]
    | list["KernelJSONValue"]
)

KERNEL_SCHEMA_MAJOR_VERSION = 1
KERNEL_SCHEMA_MINOR_VERSION = 0
KERNEL_SCHEMA_VERSION = f"{KERNEL_SCHEMA_MAJOR_VERSION}.{KERNEL_SCHEMA_MINOR_VERSION}"
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


def _parse_schema_version(schema_version: str) -> tuple[int, int]:
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("Kernel schema version must be a non-empty string")

    parts = schema_version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("Kernel schema version must use major.minor notation")
    return int(parts[0]), int(parts[1])


def _is_schema_version_compatible(schema_version: str, supported_version: str = KERNEL_SCHEMA_VERSION) -> bool:
    """Return whether a kernel schema version is compatible with a supported version."""
    try:
        request_major, request_minor = _parse_schema_version(schema_version)
        supported_major, supported_minor = _parse_schema_version(supported_version)
    except ValueError:
        return False

    return request_major == supported_major and request_minor <= supported_minor


def _validate_schema_version(schema_version: str) -> str:
    request_major, request_minor = _parse_schema_version(schema_version)
    supported_major, supported_minor = _parse_schema_version(KERNEL_SCHEMA_VERSION)
    if request_major != supported_major:
        raise ValueError(f"Unsupported kernel schema version: {schema_version}")
    if request_minor > supported_minor:
        raise ValueError(
            f"Unsupported kernel schema version: {schema_version}. "
            f"Supported version is {KERNEL_SCHEMA_VERSION}",
        )
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


def _resolve_model_class(import_path: str) -> type[Any]:
    module_name, class_name = import_path.rsplit(".", 1)
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    return model_cls


def _section_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(values or {})


def _request_section(request: KernelRequest, key: str) -> dict[str, Any]:
    value = request.payload.get(key)
    return _section_mapping(value if isinstance(value, Mapping) else None)


def _resolve_model_capability(model_key: str) -> ModelCapability:
    registry = get_model_registry()
    try:
        return registry[model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown model key: {model_key}") from exc


def _build_model_instance(model_key: str, constructor_kwargs: Mapping[str, Any] | None = None) -> tuple[ModelCapability, Any]:
    capability = _resolve_model_capability(model_key)
    model_cls = _resolve_model_class(capability.import_path)
    model = model_cls(**_section_mapping(constructor_kwargs))
    return capability, model


def _model_fit_strategy(model: Any) -> str:
    parameters = list(inspect.signature(model.fit).parameters.values())
    if parameters and parameters[0].name == "fitter":
        return "base"
    return "native"


def _model_predict_signature(model: Any) -> dict[str, bool]:
    parameters = inspect.signature(model.predict).parameters
    return {
        "y0": "y0" in parameters,
        "covariates": "covariates" in parameters,
    }


def _coerce_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _extract_observed(inputs: Mapping[str, Any]) -> np.ndarray:
    for key in ("observed", "y", "values", "adoption", "share"):
        if key in inputs:
            return _coerce_array(inputs[key])
    raise ValueError("Kernel requests require observed data in the inputs section")


def _extract_time(inputs: Mapping[str, Any]) -> np.ndarray:
    for key in ("time", "t"):
        if key in inputs:
            return _coerce_array(inputs[key])
    raise ValueError("Kernel requests require time points in the inputs section")


def _extract_predict_kwargs(
    model: Any,
    *,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    observed: np.ndarray | None = None,
) -> dict[str, Any]:
    signature_flags = _model_predict_signature(model)
    predict_kwargs: dict[str, Any] = {}
    state_predict_kwargs = _section_mapping(state.get("predict_kwargs")) if isinstance(state, Mapping) else {}

    if signature_flags["y0"]:
        y0 = inputs.get("y0", state_predict_kwargs.get("y0"))
        if y0 is None and observed is not None and observed.ndim > 1 and len(observed) > 0:
            y0 = observed[0].tolist()
        if y0 is None:
            raise ValueError(
                f"Kernel operation requires initial conditions for model '{model.__class__.__name__}'",
            )
        predict_kwargs["y0"] = y0

    if signature_flags["covariates"]:
        covariates = inputs.get("covariates", state_predict_kwargs.get("covariates"))
        if covariates is not None:
            predict_kwargs["covariates"] = covariates

    return predict_kwargs


def _call_model_predict(
    model: Any,
    time: np.ndarray,
    *,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    observed: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    predict_kwargs = _extract_predict_kwargs(model, inputs=inputs, state=state, observed=observed)
    prediction = model.predict(time, **predict_kwargs)
    return prediction, predict_kwargs


def _coerce_prediction_payload(model: Any, prediction: Any) -> KernelArrayPayload | KernelTablePayload:
    if isinstance(prediction, pd.DataFrame):
        return KernelTablePayload.from_rows(
            columns=tuple(str(column) for column in prediction.columns),
            rows=prediction.to_numpy().tolist(),
            metadata={"shape": list(prediction.shape)},
        )
    if isinstance(prediction, pd.Series):
        values = prediction.to_numpy(dtype=float)
        return KernelArrayPayload.from_values(
            values=values.tolist(),
            shape=values.shape,
            dtype=str(values.dtype),
            metadata={"shape": list(values.shape)},
        )

    array = np.asarray(prediction, dtype=float)
    if array.ndim <= 1:
        flattened = array.reshape(-1)
        shape = array.shape or (flattened.size,)
        return KernelArrayPayload.from_values(
            values=flattened.tolist(),
            shape=shape,
            dtype=str(flattened.dtype),
            metadata={"shape": list(shape)},
        )

    columns = getattr(model, "names", None)
    if columns is None:
        columns = [f"series_{index + 1}" for index in range(array.shape[1])]
    else:
        columns = [str(column) for column in columns]

    return KernelTablePayload.from_rows(
        columns=columns,
        rows=array.tolist(),
        metadata={"shape": list(array.shape)},
    )


def _compute_metrics(observed: np.ndarray, predicted: np.ndarray, n_parameters: int) -> dict[str, float]:
    observed_flat = observed.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    residuals = observed_flat - predicted_flat
    n_samples = len(observed_flat)

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((observed_flat - np.mean(observed_flat)) ** 2))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / n_samples)) if n_samples > 0 else 0.0
    mae = float(np.mean(np.abs(residuals))) if n_samples > 0 else 0.0

    if ss_res > 0 and n_samples > n_parameters:
        log_likelihood = -n_samples / 2 * (np.log(2 * np.pi) + np.log(ss_res / n_samples) + 1)
        aic = float(2 * n_parameters - 2 * log_likelihood)
        bic = float(n_parameters * np.log(n_samples) - 2 * log_likelihood)
    else:
        aic = float("inf")
        bic = float("inf")

    return {
        "MSE": float(np.mean(residuals**2)) if n_samples > 0 else 0.0,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": float(r_squared),
        "R_squared": float(r_squared),
        "RSS": ss_res,
        "AIC": aic,
        "BIC": bic,
    }


def _build_diagnostics_contract(
    model: Any,
    time: np.ndarray,
    observed: np.ndarray,
    *,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
    provenance: str = "deterministic",
    model_name: str = "",
) -> DiagnosticsContract:
    uncertainty = UncertaintySummary.point_estimate(provenance=provenance)
    warning_list: list[DiagnosticsWarning] = []

    try:
        prediction, _ = _call_model_predict(model, time, inputs=inputs, state=state, observed=observed)
        predicted = _coerce_array(prediction)
    except Exception as exc:
        warning_list.append(
            DiagnosticsWarning(
                code="prediction_failed",
                message=str(exc),
            ),
        )
        return DiagnosticsContract(
            metrics={},
            residuals=np.array([]),
            residual_analysis=None,
            warnings=warning_list,
            uncertainty=UncertaintySummary.unsupported(str(exc), provenance=provenance),
            support_level="unsupported",
            provenance=provenance,
            comparison_family="unsupported",
            model_name=model_name,
        )

    residuals = observed - predicted
    residuals_flat = residuals.reshape(-1)
    predicted_flat = predicted.reshape(-1)

    try:
        residual_analysis = analyze_residuals(residuals_flat, fitted_values=predicted_flat)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_list.append(
            DiagnosticsWarning(
                code="residual_analysis_failed",
                message=str(exc),
            ),
        )
        residual_analysis = None

    support_level = "supported" if residual_analysis is not None and uncertainty.support_level == "supported" else "partial"
    return DiagnosticsContract(
        metrics=_compute_metrics(observed, predicted, len(model.param_names) + 1),
        residuals=residuals_flat,
        residual_analysis=residual_analysis,
        warnings=warning_list,
        uncertainty=uncertainty,
        support_level=support_level,
        provenance=provenance,
        comparison_family="fitted",
        model_name=model_name,
    )


def _serialize_model_state(
    model_key: str,
    model: Any,
    *,
    constructor_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, KernelJSONValue]:
    return {
        "model_key": model_key,
        "model_name": model.__class__.__name__,
        "constructor_kwargs": _as_dict_value(_section_mapping(constructor_kwargs)),
        "parameters": _as_dict_value(_section_mapping(getattr(model, "params_", {}))),
        "predict_kwargs": _as_dict_value(_section_mapping(predict_kwargs)),
    }


def _extract_model_state(
    request: KernelRequest,
    *,
    allow_missing_parameters: bool = False,
) -> tuple[KernelJSONValue | None, Any, Mapping[str, Any], dict[str, Any]]:
    inputs = _request_section(request, "inputs")
    state = _request_section(request, "state")
    constructor_kwargs = _section_mapping(
        request.payload.get("model_kwargs")
        if isinstance(request.payload.get("model_kwargs"), Mapping)
        else request.payload.get("constructor_kwargs")
        if isinstance(request.payload.get("constructor_kwargs"), Mapping)
        else state.get("constructor_kwargs") if isinstance(state, Mapping) else {},
    )

    capability, model = _build_model_instance(request.model_key or "", constructor_kwargs)

    state_model_key = state.get("model_key", request.model_key) if isinstance(state, Mapping) else request.model_key
    if state_model_key != request.model_key:
        raise ValueError(
            f"Kernel request model_key '{request.model_key}' does not match state model_key '{state_model_key}'",
        )

    parameters = request.payload.get("parameters")
    if not isinstance(parameters, Mapping) and isinstance(state, Mapping):
        parameters = state.get("parameters")
    if isinstance(parameters, Mapping) and parameters:
        model.params_ = _section_mapping(parameters)
    elif not allow_missing_parameters and request.operation != KernelOperation.FIT_MODEL.value:
        raise ValueError("Kernel requests for model execution require fitted parameters in state or parameters")

    return capability, model, inputs, constructor_kwargs


def _kernel_error_response(
    operation: str,
    code: KernelErrorCode,
    message: str,
    *,
    details: Mapping[str, KernelJSONValue] | None = None,
    retryable: bool = False,
    metadata: Mapping[str, KernelJSONValue] | None = None,
) -> KernelResponse:
    return KernelResponse(
        operation=operation,
        error=KernelError(
            code=code.value,
            message=message,
            operation=operation,
            details=_section_mapping(details),
            retryable=retryable,
        ),
        metadata=_section_mapping(metadata),
    )


def _kernel_success_response(
    operation: str,
    result: Any,
    *,
    metadata: Mapping[str, KernelJSONValue] | None = None,
) -> KernelResponse:
    return KernelResponse(operation=operation, result=result, metadata=_section_mapping(metadata))


def fit_model(request: KernelRequest) -> KernelResponse:
    """Fit a stable model using the kernel adapter surface."""
    try:
        inputs = _request_section(request, "inputs")
        time = _extract_time(inputs)
        observed = _extract_observed(inputs)
        model_kwargs = _section_mapping(
            request.payload.get("model_kwargs")
            if isinstance(request.payload.get("model_kwargs"), Mapping)
            else request.payload.get("constructor_kwargs")
            if isinstance(request.payload.get("constructor_kwargs"), Mapping)
            else None,
        )
        fit_options = _section_mapping(request.payload.get("fit_options") if isinstance(request.payload.get("fit_options"), Mapping) else None)
        fitter_options = _section_mapping(
            request.payload.get("fitter_options") if isinstance(request.payload.get("fitter_options"), Mapping) else None,
        )

        capability, model = _build_model_instance(request.model_key or "", model_kwargs)
        fit_strategy = _model_fit_strategy(model)

        if fit_strategy == "base":
            fitter_options = dict(fitter_options)
            fitter_options.setdefault("method", "curve_fit")
            fitter = ScipyFitter(**fitter_options)
            model.fit(fitter, time, observed, **fit_options)
        else:
            native_fit_kwargs: dict[str, Any] = dict(fit_options)
            signature = inspect.signature(model.fit).parameters
            if "covariates" in signature and "covariates" in inputs:
                native_fit_kwargs["covariates"] = inputs["covariates"]
            model.fit(time, observed, **native_fit_kwargs)

        predict_kwargs = _extract_predict_kwargs(model, inputs=inputs, observed=observed)
        diagnostics = _build_diagnostics_contract(
            model,
            time,
            observed,
            inputs=inputs,
            state=None,
            provenance="deterministic",
            model_name=model.__class__.__name__,
        )
        prediction, _ = _call_model_predict(model, time, inputs=inputs, observed=observed)
        prediction_payload = _coerce_prediction_payload(model, prediction)
        state = _serialize_model_state(
            request.model_key or "",
            model,
            constructor_kwargs=model_kwargs,
            predict_kwargs=predict_kwargs,
        )

        return _kernel_success_response(
            request.operation,
            {
                "model_key": request.model_key,
                "model_name": model.__class__.__name__,
                "family": capability.family,
                "parameters": _as_dict_value(getattr(model, "params_", {})),
                "predictions": prediction_payload.to_dict(),
                "diagnostics": diagnostics.to_dict(),
                "state": state,
            },
            metadata={
                "model_key": request.model_key,
                "family": capability.family,
                "support_level": diagnostics.support_level,
            },
        )
    except Exception as exc:
        return _kernel_error_response(
            request.operation,
            KernelErrorCode.INVALID_REQUEST,
            str(exc),
            metadata={"model_key": request.model_key},
        )


def predict_model(request: KernelRequest) -> KernelResponse:
    """Run predictions for a fitted stable model."""
    try:
        capability, model, inputs, constructor_kwargs = _extract_model_state(request)
        state = _request_section(request, "state")
        time = _extract_time(inputs)
        observed = _coerce_array(inputs["observed"]) if "observed" in inputs else None
        prediction, _ = _call_model_predict(model, time, inputs=inputs, state=state, observed=observed)
        payload = _coerce_prediction_payload(model, prediction)
        return _kernel_success_response(
            request.operation,
            payload,
            metadata={"model_key": request.model_key, "family": capability.family, "model_name": model.__class__.__name__},
        )
    except Exception as exc:
        return _kernel_error_response(
            request.operation,
            KernelErrorCode.INVALID_REQUEST,
            str(exc),
            metadata={"model_key": request.model_key},
        )


def simulate_model(request: KernelRequest) -> KernelResponse:
    """Simulate a stable model using the same execution path as prediction."""
    try:
        capability, model, inputs, _ = _extract_model_state(request)
        state = _request_section(request, "state")
        time = _extract_time(inputs)
        observed = _coerce_array(inputs["observed"]) if "observed" in inputs else None
        prediction, _ = _call_model_predict(model, time, inputs=inputs, state=state, observed=observed)
        payload = _coerce_prediction_payload(model, prediction)
        return _kernel_success_response(
            request.operation,
            payload,
            metadata={"model_key": request.model_key, "family": capability.family, "model_name": model.__class__.__name__},
        )
    except Exception as exc:
        return _kernel_error_response(
            request.operation,
            KernelErrorCode.INVALID_REQUEST,
            str(exc),
            metadata={"model_key": request.model_key},
        )


def summarize_model(request: KernelRequest) -> KernelResponse:
    """Summarize a fitted stable model and return diagnostics when data are available."""
    try:
        capability, model, inputs, constructor_kwargs = _extract_model_state(request)
        state = _request_section(request, "state")
        time = _extract_time(inputs) if "time" in inputs or "t" in inputs else None
        observed = _extract_observed(inputs) if any(key in inputs for key in ("observed", "y", "values", "adoption", "share")) else None

        diagnostics: DiagnosticsContract | None = None
        if time is not None and observed is not None:
            diagnostics = _build_diagnostics_contract(
                model,
                time,
                observed,
                inputs=inputs,
                state=state,
                provenance="deterministic",
                model_name=model.__class__.__name__,
            )

        result = {
            "model_key": request.model_key,
            "model_name": model.__class__.__name__,
            "family": capability.family,
            "parameter_names": list(getattr(model, "param_names", ())),
            "parameters": _as_dict_value(getattr(model, "params_", {})),
            "constructor_kwargs": _as_dict_value(constructor_kwargs),
            "state": _serialize_model_state(
                request.model_key or "",
                model,
                constructor_kwargs=constructor_kwargs,
                predict_kwargs=_section_mapping(state.get("predict_kwargs")) if isinstance(state, Mapping) else None,
            ),
        }
        if diagnostics is not None:
            result["diagnostics"] = diagnostics.to_dict()

        return _kernel_success_response(
            request.operation,
            result,
            metadata={"model_key": request.model_key, "family": capability.family, "model_name": model.__class__.__name__},
        )
    except Exception as exc:
        return _kernel_error_response(
            request.operation,
            KernelErrorCode.INVALID_REQUEST,
            str(exc),
            metadata={"model_key": request.model_key},
        )


def diagnose_model(request: KernelRequest) -> KernelResponse:
    """Return a structured diagnostics contract for a fitted stable model."""
    try:
        capability, model, inputs, _ = _extract_model_state(request)
        state = _request_section(request, "state")
        time = _extract_time(inputs)
        observed = _extract_observed(inputs)
        diagnostics = _build_diagnostics_contract(
            model,
            time,
            observed,
            inputs=inputs,
            state=state,
            provenance="deterministic",
            model_name=model.__class__.__name__,
        )
        return _kernel_success_response(
            request.operation,
            {
                "diagnostics": diagnostics.to_dict(),
                "state": _serialize_model_state(
                    request.model_key or "",
                    model,
                    constructor_kwargs=_section_mapping(state.get("constructor_kwargs")) if isinstance(state, Mapping) else None,
                    predict_kwargs=_section_mapping(state.get("predict_kwargs")) if isinstance(state, Mapping) else None,
                ),
            },
            metadata={"model_key": request.model_key, "family": capability.family, "model_name": model.__class__.__name__},
        )
    except Exception as exc:
        return _kernel_error_response(
            request.operation,
            KernelErrorCode.INVALID_REQUEST,
            str(exc),
            metadata={"model_key": request.model_key},
        )


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
