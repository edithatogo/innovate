"""Remote execution contract and local in-process adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast

from . import kernel

RemoteExecutionStatus = Literal["ok", "error"]


def _copy_metadata(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(values or {})


@dataclass(frozen=True, slots=True)
class RemoteExecutionContext:
    """Correlation, tenancy, and principal context for remote execution."""

    request_id: str
    tenant_id: str
    principal: str
    trace_id: str = ""
    auth_scope: str = ""
    data_retention: str = "ephemeral"

    def __post_init__(self) -> None:
        for field_name in ("request_id", "tenant_id", "principal"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Remote execution {field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Serialize the context to a JSON-compatible dictionary."""
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "principal": self.principal,
            "trace_id": self.trace_id,
            "auth_scope": self.auth_scope,
            "data_retention": self.data_retention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RemoteExecutionContext:
        """Build context from a JSON-compatible dictionary."""
        return cls(
            request_id=str(data["request_id"]),
            tenant_id=str(data["tenant_id"]),
            principal=str(data["principal"]),
            trace_id=str(data.get("trace_id", "")),
            auth_scope=str(data.get("auth_scope", "")),
            data_retention=str(data.get("data_retention", "ephemeral")),
        )


@dataclass(frozen=True, slots=True)
class RemoteExecutionPolicy:
    """Local policy used by hosted adapters before dispatching kernel requests."""

    eligible_operations: tuple[str, ...] = (
        kernel.KernelOperation.DISCOVER_MODELS.value,
        kernel.KernelOperation.PREDICT_MODEL.value,
        kernel.KernelOperation.SIMULATE_MODEL.value,
        kernel.KernelOperation.SUMMARIZE_MODEL.value,
        kernel.KernelOperation.DIAGNOSE_MODEL.value,
    )
    local_only_by_default: tuple[str, ...] = (kernel.KernelOperation.FIT_MODEL.value,)
    required_auth_scope: str = field(
        default_factory=lambda: os.getenv("INNOVATE_REQUIRED_AUTH_SCOPE", "UNCONFIGURED_DENY_ALL")
    )
    max_payload_bytes: int = 1_000_000
    data_retention: str = "ephemeral"

    def allows(self, request: kernel.KernelRequest, context: RemoteExecutionContext) -> tuple[bool, str]:
        """Return whether a request is allowed and the denial reason when blocked."""
        if context.auth_scope != self.required_auth_scope:
            return False, "remote execution auth_scope is not authorized"
        if request.operation not in self.eligible_operations:
            return False, f"remote execution does not allow operation '{request.operation}'"
        return True, ""


@dataclass(frozen=True, slots=True)
class RemoteExecutionRequest:
    """Remote execution envelope that carries a kernel request and context."""

    kernel_request: kernel.KernelRequest
    context: RemoteExecutionContext
    schema_version: str = kernel.KERNEL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != kernel.KERNEL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported remote execution schema version: {self.schema_version}")
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the remote request envelope."""
        return {
            "schema_version": self.schema_version,
            "kernel_request": self.kernel_request.to_dict(),
            "context": self.context.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RemoteExecutionRequest:
        """Build a remote request from a JSON-compatible dictionary."""
        return cls(
            schema_version=str(data.get("schema_version", kernel.KERNEL_SCHEMA_VERSION)),
            kernel_request=kernel.KernelRequest.from_dict(data["kernel_request"]),
            context=RemoteExecutionContext.from_dict(data["context"]),
            metadata=_copy_metadata(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}),
        )


@dataclass(frozen=True, slots=True)
class RemoteExecutionResponse:
    """Remote execution result with kernel response, provenance, and observability."""

    kernel_response: kernel.KernelResponse
    status: RemoteExecutionStatus
    provenance: dict[str, Any]
    observability: dict[str, Any]
    schema_version: str = kernel.KERNEL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the remote execution response."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "kernel_response": self.kernel_response.to_dict(),
            "provenance": dict(self.provenance),
            "observability": dict(self.observability),
        }


def _dispatch_kernel_request(request: kernel.KernelRequest) -> kernel.KernelResponse:
    if request.operation == kernel.KernelOperation.DISCOVER_MODELS.value:
        discovery = kernel.discover_models()
        return kernel.KernelResponse(
            operation=request.operation,
            result=cast(dict[str, kernel.KernelJSONValue], discovery.to_dict()),
            metadata={"runtime": "python", "backend": "registry"},
        )

    dispatch = {
        kernel.KernelOperation.FIT_MODEL.value: kernel.fit_model,
        kernel.KernelOperation.PREDICT_MODEL.value: kernel.predict_model,
        kernel.KernelOperation.SIMULATE_MODEL.value: kernel.simulate_model,
        kernel.KernelOperation.SUMMARIZE_MODEL.value: kernel.summarize_model,
        kernel.KernelOperation.DIAGNOSE_MODEL.value: kernel.diagnose_model,
    }
    return dispatch[request.operation](request)


class InProcessRemoteExecutor:
    """Minimal local adapter that exercises the remote execution contract."""

    def __init__(
        self,
        *,
        policy: RemoteExecutionPolicy | None = None,
        runtime: str = "python",
        backend: str = "numpy_scipy",
    ) -> None:
        self.policy = policy or RemoteExecutionPolicy()
        self.runtime = runtime
        self.backend = backend

    def execute(self, request: RemoteExecutionRequest) -> RemoteExecutionResponse:
        """Execute a remote request through the local functional kernel."""
        started = perf_counter()
        allowed, reason = self.policy.allows(request.kernel_request, request.context)
        if not allowed:
            response = kernel.KernelResponse(
                operation=request.kernel_request.operation,
                error=kernel.KernelError(
                    code=kernel.KernelErrorCode.UNSUPPORTED_OPERATION.value,
                    message=reason,
                    operation=request.kernel_request.operation,
                    details={
                        "request_id": request.context.request_id,
                        "tenant_id": request.context.tenant_id,
                    },
                ),
            )
            return self._wrap_response(request, response, started=started, status="error")

        response = _dispatch_kernel_request(request.kernel_request)
        status: RemoteExecutionStatus = "error" if response.error is not None else "ok"
        return self._wrap_response(request, response, started=started, status=status)

    def _wrap_response(
        self,
        request: RemoteExecutionRequest,
        response: kernel.KernelResponse,
        *,
        started: float,
        status: RemoteExecutionStatus,
    ) -> RemoteExecutionResponse:
        elapsed_ms = (perf_counter() - started) * 1000
        observability = {
            "request_id": request.context.request_id,
            "tenant_id": request.context.tenant_id,
            "principal": request.context.principal,
            "trace_id": request.context.trace_id,
            "operation": request.kernel_request.operation,
            "duration_ms": elapsed_ms,
            "status": status,
        }
        provenance = {
            "execution_location": "in_process",
            "runtime": self.runtime,
            "backend": self.backend,
            "backend_provenance": "NumPy/SciPy reference path",
            "xla": {
                "used": self.runtime == "JAX/XLA",
                "public_contract": False,
            },
            "rust_native": self.runtime == "rust_native",
            "bridge_fallback": self.runtime == "bridge_fallback",
            "data_retention": request.context.data_retention,
        }
        return RemoteExecutionResponse(
            kernel_response=response,
            status=status,
            provenance=provenance,
            observability=observability,
        )


def describe_remote_execution_contract() -> dict[str, Any]:
    """Describe remote execution boundaries, security, and observability requirements."""
    policy = RemoteExecutionPolicy()
    return {
        "schema_version": kernel.KERNEL_SCHEMA_VERSION,
        "eligible_operations": policy.eligible_operations,
        "local_only_by_default": policy.local_only_by_default,
        "request_fields": (
            "schema_version",
            "kernel_request",
            "context",
            "metadata",
        ),
        "response_fields": (
            "schema_version",
            "status",
            "kernel_response",
            "provenance",
            "observability",
        ),
        "security": {
            "authentication": "required before hosted deployment",
            "authorization": "required auth_scope and operation allow-list",
            "tenant_isolation": "tenant_id must be logged and enforced by hosted adapters",
            "data_retention": "ephemeral by default; persisted artifacts require explicit policy",
            "blocked_patterns": (
                "arbitrary remote code execution",
                "unversioned request schemas",
                "public exposure of XLA internals",
            ),
        },
        "observability": {
            "required_fields": (
                "request_id",
                "tenant_id",
                "principal",
                "trace_id",
                "operation",
                "duration_ms",
                "status",
            ),
            "signals": ("structured_logs", "traces", "metrics"),
        },
        "backend_provenance": {
            "supported_runtimes": ("NumPy/SciPy", "JAX/XLA", "Rust-native", "bridge fallback"),
            "public_contract": "kernel schema and Arrow-compatible payloads",
        },
    }


__all__ = [
    "InProcessRemoteExecutor",
    "RemoteExecutionContext",
    "RemoteExecutionPolicy",
    "RemoteExecutionRequest",
    "RemoteExecutionResponse",
    "describe_remote_execution_contract",
]
