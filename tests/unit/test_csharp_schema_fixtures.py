"""Schema fixture compatibility checks for the C# thin binding."""

from __future__ import annotations

import json
from pathlib import Path

from innovate import kernel

FIXTURE_ROOT = Path("bindings/csharp/Innovate.Kernel.Tests/fixtures")


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_csharp_request_fixture_is_accepted_by_functional_kernel_schema() -> None:
    """The C# request fixture should remain valid for the Python kernel contract."""
    request = kernel.KernelRequest.from_dict(_load_fixture("kernel_request.predict_model.json"))

    assert request.schema_version == kernel.KERNEL_SCHEMA_VERSION
    assert request.operation == kernel.KernelOperation.PREDICT_MODEL.value
    assert request.model_key == "bass"
    assert request.payload["t"]["dtype"] == "float64"  # type: ignore[index]


def test_csharp_response_fixtures_are_accepted_by_functional_kernel_schema() -> None:
    """The C# success and error fixtures should match the shared response envelope."""
    response = kernel.KernelResponse.from_dict(_load_fixture("kernel_response.predict_model.json"))
    error_response = kernel.KernelResponse.from_dict(_load_fixture("kernel_response.error.json"))

    assert response.schema_version == kernel.KERNEL_SCHEMA_VERSION
    assert response.operation == kernel.KernelOperation.PREDICT_MODEL.value
    assert response.error is None
    assert response.result is not None

    assert error_response.error is not None
    assert error_response.error.code == kernel.KernelErrorCode.INVALID_PAYLOAD.value
    assert error_response.error.operation == kernel.KernelOperation.PREDICT_MODEL.value


def test_csharp_operation_constants_match_functional_kernel_contract() -> None:
    """The C# operation constants should mirror the functional kernel operation list."""
    source = Path("bindings/csharp/Innovate.Kernel/KernelOperation.cs").read_text(encoding="utf-8")

    for operation in kernel.KERNEL_OPERATIONS:
        assert f'= "{operation}"' in source
