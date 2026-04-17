"""Tests for the functional kernel contract surface."""

from __future__ import annotations

import pytest


def test_kernel_contract_exposes_versioned_operations() -> None:
    """The kernel surface should publish a stable, versioned operation list."""
    from innovate import kernel

    assert kernel.KERNEL_SCHEMA_VERSION == "1.0"
    assert kernel.KERNEL_OPERATIONS == (
        "discover_models",
        "fit_model",
        "predict_model",
        "simulate_model",
        "summarize_model",
        "diagnose_model",
    )

    request = kernel.KernelRequest(
        operation="discover_models",
        model_key=None,
        payload={"constraints": {"family": "diffusion"}},
    )

    assert request.to_dict() == {
        "schema_version": "1.0",
        "operation": "discover_models",
        "model_key": None,
        "payload": {"constraints": {"family": "diffusion"}},
        "metadata": {},
    }


def test_kernel_contract_round_trips_request_response_and_errors() -> None:
    """Kernel envelopes should round-trip through JSON-friendly payloads."""
    from innovate import kernel

    request = kernel.KernelRequest(
        schema_version="1.0",
        operation="fit_model",
        model_key="bass",
        payload={
            "inputs": {
                "time": [1.0, 2.0, 3.0],
                "adoption": [10.0, 20.0, 35.0],
            },
            "fit_options": {"backend": "numpy"},
        },
        metadata={"request_id": "req-1"},
    )
    response = kernel.KernelResponse(
        operation="fit_model",
        result={
            "parameters": {"p": 0.02, "q": 0.38, "m": 1000.0},
            "metadata": {"model_key": "bass"},
        },
        metadata={"request_id": "req-1"},
    )
    error = kernel.KernelError(
        code="invalid_request",
        message="Expected one-dimensional time series.",
        operation="predict_model",
        details={"field": "time"},
    )

    assert kernel.KernelRequest.from_dict(request.to_dict()) == request
    assert kernel.KernelResponse.from_dict(response.to_dict()) == response
    assert kernel.KernelError.from_dict(error.to_dict()) == error

    serialized = kernel.KernelResponse(
        operation="fit_model",
        result={"ok": True},
        metadata={
            "request_id": "req-1",
            "error_code": kernel.KernelErrorCode.INVALID_REQUEST,
            "tags": ("fit", "roundtrip"),
        },
    )
    assert serialized.to_dict()["metadata"] == {
        "request_id": "req-1",
        "error_code": "invalid_request",
        "tags": ["fit", "roundtrip"],
    }


def test_kernel_contract_distinguishes_arrays_and_tabular_payloads() -> None:
    """Tabular payloads should preserve columnar structure and metadata."""
    from innovate import kernel

    array_payload = kernel.KernelArrayPayload.from_values(
        values=(1.0, 2.0, 3.0),
        shape=(3,),
        dtype="float64",
    )
    table_payload = kernel.KernelTablePayload.from_rows(
        columns=("time", "adoption"),
        rows=((1.0, 10.0), (2.0, 22.0)),
        metadata={"source": "synthetic"},
    )

    assert array_payload.to_dict() == {
        "shape": [3],
        "dtype": "float64",
        "values": [1.0, 2.0, 3.0],
        "metadata": {},
    }
    assert table_payload.to_dict() == {
        "columns": ["time", "adoption"],
        "rows": [[1.0, 10.0], [2.0, 22.0]],
        "metadata": {"source": "synthetic"},
    }


def test_kernel_contract_rejects_unknown_operations() -> None:
    """Unknown operations should fail fast with a stable error code."""
    from innovate import kernel

    with pytest.raises(ValueError, match="Unknown kernel operation"):
        kernel.KernelRequest(operation="train_model", model_key="bass", payload={})


def test_kernel_contract_validates_kernel_version_and_payload_shapes() -> None:
    """Validation should reject malformed schema versions and payload shapes."""
    from innovate import kernel

    with pytest.raises(ValueError, match="major.minor notation"):
        kernel._validate_schema_version("1")

    with pytest.raises(ValueError, match="Unsupported kernel schema version"):
        kernel._validate_schema_version("2.0")

    with pytest.raises(ValueError, match="non-empty string"):
        kernel.KernelError(code="", message="missing code")

    with pytest.raises(ValueError, match="non-empty string"):
        kernel.KernelError(code="invalid_request", message="")

    with pytest.raises(ValueError, match="Unknown kernel operation"):
        kernel.KernelError(code="invalid_request", message="bad op", operation="train_model")

    with pytest.raises(ValueError, match="Kernel arrays must declare a shape"):
        kernel.KernelArrayPayload(shape=(), dtype="float64", values=())

    with pytest.raises(ValueError, match="non-empty string"):
        kernel.KernelArrayPayload(shape=(1,), dtype="", values=(1.0,))

    with pytest.raises(ValueError, match="must match the number of values"):
        kernel.KernelArrayPayload(shape=(2,), dtype="float64", values=(1.0,))

    with pytest.raises(ValueError, match="at least one column"):
        kernel.KernelTablePayload(columns=(), rows=())

    with pytest.raises(ValueError, match="non-empty strings"):
        kernel.KernelTablePayload(columns=("time", ""), rows=((1.0, 2.0),))

    with pytest.raises(ValueError, match="must match the number of columns"):
        kernel.KernelTablePayload(columns=("time", "adoption"), rows=((1.0,),))

    with pytest.raises(ValueError, match="requires a model_key"):
        kernel.KernelRequest(operation="fit_model", model_key=None, payload={})

    with pytest.raises(TypeError, match="Kernel response errors must be KernelError instances"):
        kernel.KernelResponse(operation="fit_model", result={}, error="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="either a result or an error"):
        kernel.KernelResponse(operation="fit_model")


def test_kernel_contract_discovery_and_stub_operations_behave_as_documented() -> None:
    """Discovery should reflect the capability registry and stubs should fail clearly."""
    from innovate import kernel
    from innovate.capabilities import get_model_registry

    discovery = kernel.discover_models()
    capability = next(iter(get_model_registry().values()))
    record = kernel.KernelDiscoveryRecord.from_capability(capability)

    assert discovery.schema_version == "1.0"
    assert discovery.models
    assert discovery.models[0].to_dict()["key"] in {record.key for record in discovery.models}
    assert kernel.list_kernel_operations() == kernel.KERNEL_OPERATIONS
    assert kernel.KernelDiscoveryResponse.from_dict(discovery.to_dict()) == discovery
    assert record.to_dict() == next(model.to_dict() for model in discovery.models if model.key == record.key)

    request = kernel.KernelRequest(
        operation="fit_model",
        model_key="bass",
        payload={"inputs": {"time": [1.0, 2.0], "adoption": [4.0, 9.0]}},
    )

    with pytest.raises(NotImplementedError, match="fit_model"):
        kernel.fit_model(request)
    with pytest.raises(NotImplementedError, match="predict_model"):
        kernel.predict_model(request)
    with pytest.raises(NotImplementedError, match="simulate_model"):
        kernel.simulate_model(request)
    with pytest.raises(NotImplementedError, match="summarize_model"):
        kernel.summarize_model(request)
    with pytest.raises(NotImplementedError, match="diagnose_model"):
        kernel.diagnose_model(request)


def test_kernel_contract_round_trips_array_and_table_response_payloads() -> None:
    """Kernel responses should preserve structured payload variants."""
    from innovate import kernel

    array_response = kernel.KernelResponse(
        operation="predict_model",
        result=kernel.KernelArrayPayload.from_values(
            values=(1.0, 2.0, 3.0),
            shape=(3,),
            dtype="float64",
        ),
        metadata={"request_id": "req-array"},
    )
    table_response = kernel.KernelResponse(
        operation="diagnose_model",
        result=kernel.KernelTablePayload.from_rows(
            columns=("time", "residual"),
            rows=((1.0, 0.1), (2.0, -0.2)),
            metadata={"source": "synthetic"},
        ),
        metadata={"request_id": "req-table"},
    )

    assert kernel.KernelResponse.from_dict(array_response.to_dict()) == array_response
    assert kernel.KernelResponse.from_dict(table_response.to_dict()) == table_response


def test_kernel_contract_round_trips_error_only_response_and_bare_error() -> None:
    """Kernel errors should round-trip whether or not they carry an operation."""
    from innovate import kernel

    bare_error = kernel.KernelError(code="invalid_request", message="missing payload")
    error_response = kernel.KernelResponse(
        operation="predict_model",
        error=kernel.KernelError(
            code="invalid_payload",
            message="Expected a numeric feature vector.",
            operation="predict_model",
        ),
    )

    assert kernel.KernelError.from_dict(bare_error.to_dict()) == bare_error
    assert kernel.KernelResponse.from_dict(error_response.to_dict()) == error_response


def test_kernel_contract_rejects_blank_schema_versions() -> None:
    """Schema validation should reject empty and malformed versions."""
    from innovate import kernel

    with pytest.raises(ValueError, match="non-empty string"):
        kernel._validate_schema_version("")
