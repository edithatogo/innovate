"""Tests for the hosted-service and remote-execution contract."""

from __future__ import annotations

from innovate import kernel
from innovate.remote_execution import (
    InProcessRemoteExecutor,
    RemoteExecutionContext,
    RemoteExecutionPolicy,
    RemoteExecutionRequest,
    describe_remote_execution_contract,
)


def test_remote_execution_contract_documents_boundaries_and_risk_controls() -> None:
    """The remote contract should document eligibility, security, and observability."""
    contract = describe_remote_execution_contract()

    assert contract["schema_version"] == kernel.KERNEL_SCHEMA_VERSION
    assert set(contract["eligible_operations"]) >= {"discover_models", "predict_model", "simulate_model"}
    assert "fit_model" in contract["local_only_by_default"]
    assert "authorization" in contract["security"]
    assert "tenant_id" in contract["observability"]["required_fields"]
    assert "JAX/XLA" in contract["backend_provenance"]["supported_runtimes"]


def test_in_process_remote_executor_preserves_kernel_schema_and_correlation() -> None:
    """The local adapter should wrap kernel responses without changing the kernel ABI."""
    executor = InProcessRemoteExecutor()
    request = RemoteExecutionRequest(
        kernel_request=kernel.KernelRequest(
            operation=kernel.KernelOperation.DISCOVER_MODELS.value,
            model_key=None,
            payload={},
        ),
        context=RemoteExecutionContext(
            request_id="req-001",
            tenant_id="tenant-a",
            principal="service-user",
            trace_id="trace-001",
        ),
    )

    response = executor.execute(request)
    payload = response.to_dict()

    assert response.kernel_response.schema_version == kernel.KERNEL_SCHEMA_VERSION
    assert response.status == "ok"
    assert response.observability["request_id"] == "req-001"
    assert response.provenance["execution_location"] == "in_process"
    assert response.provenance["runtime"] == "python"
    assert payload["kernel_response"]["schema_version"] == kernel.KERNEL_SCHEMA_VERSION
    assert payload["observability"]["trace_id"] == "trace-001"


def test_remote_executor_returns_structured_error_for_disallowed_operation() -> None:
    """Remote policy failures should be structured and language-binding friendly."""
    executor = InProcessRemoteExecutor(
        policy=RemoteExecutionPolicy(eligible_operations=("discover_models",)),
    )
    request = RemoteExecutionRequest(
        kernel_request=kernel.KernelRequest(
            operation=kernel.KernelOperation.FIT_MODEL.value,
            model_key="bass",
            payload={},
        ),
        context=RemoteExecutionContext(
            request_id="req-denied",
            tenant_id="tenant-a",
            principal="service-user",
        ),
    )

    response = executor.execute(request)

    assert response.status == "error"
    assert response.kernel_response.error is not None
    assert response.kernel_response.error.code == kernel.KernelErrorCode.UNSUPPORTED_OPERATION.value
    assert response.kernel_response.error.details["request_id"] == "req-denied"
    assert response.observability["request_id"] == "req-denied"


def test_remote_execution_request_round_trips_from_dict() -> None:
    """Remote request envelopes should be stable JSON-compatible dictionaries."""
    request = RemoteExecutionRequest(
        kernel_request=kernel.KernelRequest(
            operation=kernel.KernelOperation.DISCOVER_MODELS.value,
            model_key=None,
            payload={},
        ),
        context=RemoteExecutionContext(
            request_id="req-roundtrip",
            tenant_id="tenant-a",
            principal="service-user",
        ),
    )

    restored = RemoteExecutionRequest.from_dict(request.to_dict())

    assert restored == request
