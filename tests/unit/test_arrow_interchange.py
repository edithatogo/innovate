"""Tests for the Arrow interchange surface."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest


def test_arrow_interchange_describes_the_versioned_contract() -> None:
    """The interchange module should publish a single versioned contract summary."""
    from innovate import arrow_interchange

    spec = arrow_interchange.describe_arrow_interchange()

    assert spec["schema_version"] == "1.0"
    assert spec["payloads"]["array"]["metadata_keys"] == (
        "innovate.kind",
        "innovate.schema_version",
        "innovate.shape",
        "innovate.dtype",
    )
    assert spec["payloads"]["table"]["metadata_keys"] == (
        "innovate.kind",
        "innovate.schema_version",
        "innovate.columns",
    )
    assert spec["payloads"]["discovery"]["metadata_keys"] == (
        "innovate.kind",
        "innovate.schema_version",
    )


def test_kernel_array_payload_round_trips_through_arrow_and_pandas() -> None:
    """Kernel array payloads should round-trip through Arrow and pandas helpers."""
    from innovate import arrow_interchange, kernel

    payload = kernel.KernelArrayPayload.from_values(
        values=(1.0, 2.0, 3.0, 4.0),
        shape=(2, 2),
        dtype="float64",
        metadata={"source": "synthetic"},
    )

    arrow_table = arrow_interchange.kernel_array_payload_to_table(payload)

    assert isinstance(arrow_table, pa.Table)
    assert arrow_table.num_columns == 1
    assert arrow_table.column_names == ["value"]
    assert arrow_table.schema.metadata[b"innovate.kind"] == b"array"
    assert arrow_table.schema.metadata[b"innovate.schema_version"] == b"1.0"

    restored = arrow_interchange.kernel_array_payload_from_table(arrow_table)
    assert restored == payload

    frame = arrow_interchange.kernel_array_payload_to_dataframe(payload)
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == ["value"]
    assert frame.to_dict(orient="list") == {"value": [1.0, 2.0, 3.0, 4.0]}

    from_frame = arrow_interchange.kernel_array_payload_from_dataframe(
        frame,
        shape=(2, 2),
        dtype="float64",
        metadata={"source": "synthetic"},
    )
    assert from_frame == payload


def test_kernel_table_payload_round_trips_through_arrow_and_pandas() -> None:
    """Kernel table payloads should preserve columns, rows, and metadata."""
    from innovate import arrow_interchange, kernel

    payload = kernel.KernelTablePayload.from_rows(
        columns=("time", "adoption"),
        rows=((1.0, 10.0), (2.0, 22.0)),
        metadata={"source": "synthetic"},
    )

    arrow_table = arrow_interchange.kernel_table_payload_to_table(payload)

    assert isinstance(arrow_table, pa.Table)
    assert arrow_table.column_names == ["time", "adoption"]
    assert arrow_table.schema.metadata[b"innovate.kind"] == b"table"
    assert arrow_table.schema.metadata[b"innovate.schema_version"] == b"1.0"

    restored = arrow_interchange.kernel_table_payload_from_table(arrow_table)
    assert restored == payload

    frame = arrow_interchange.kernel_table_payload_to_dataframe(payload)
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == ["time", "adoption"]
    assert frame.to_dict(orient="records") == [
        {"time": 1.0, "adoption": 10.0},
        {"time": 2.0, "adoption": 22.0},
    ]

    from_frame = arrow_interchange.kernel_table_payload_from_dataframe(
        frame,
        metadata={"source": "synthetic"},
    )
    assert from_frame == payload


def test_diagnostics_artifact_tables_round_trip_through_arrow() -> None:
    """Diagnostics artifacts should use the existing Arrow table contract."""
    import numpy as np

    from innovate import arrow_interchange
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.diagnostics_contract import build_diagnostics_contract

    t = np.linspace(1, 6, 6)
    model = BassModel()
    model.params_ = {"p": 0.03, "q": 0.35, "m": 1000.0}
    y = model.predict(t) + np.array([0.0, 0.1, -0.05, 0.08, -0.02, 0.03])
    artifact = build_diagnostics_contract(model, t, y, model_name="BassModel").to_artifact_payload()

    residual_table_payload = artifact.to_table_payloads()["residuals"]
    arrow_table = arrow_interchange.kernel_table_payload_to_table(residual_table_payload)

    assert arrow_table.schema.metadata[b"innovate.kind"] == b"table"
    assert arrow_table.schema.metadata[b"diagnostics_artifact"] == b'"residuals"'
    assert arrow_table.schema.metadata[b"diagnostics_artifact_schema_version"] == b'"1.0"'

    restored = arrow_interchange.kernel_table_payload_from_table(arrow_table)
    assert restored == residual_table_payload


def test_kernel_discovery_response_round_trips_through_arrow() -> None:
    """Model metadata should round-trip as an Arrow table as well."""
    from innovate import arrow_interchange, kernel

    discovery = kernel.KernelDiscoveryResponse(
        models=(
            kernel.KernelDiscoveryRecord(
                key="bass",
                family="diffusion",
                import_path="innovate.diffuse.bass.BassModel",
                stability="stable",
                supports_covariates=False,
                supports_multivariate_output=False,
                supported_backends=("numpy",),
                optional_dependencies=(),
                supports_simulation=True,
                supports_summarize=True,
            ),
        ),
        metadata={"source": "registry"},
    )

    arrow_table = arrow_interchange.kernel_discovery_response_to_table(discovery)
    assert isinstance(arrow_table, pa.Table)
    assert arrow_table.column_names == [
        "key",
        "family",
        "import_path",
        "stability",
        "supports_covariates",
        "supports_multivariate_output",
        "supported_backends",
        "optional_dependencies",
        "supports_simulation",
        "supports_summarize",
    ]

    restored = arrow_interchange.kernel_discovery_response_from_table(arrow_table)
    assert restored == discovery


def test_arrow_interchange_rejects_tables_without_contract_metadata() -> None:
    """Helpers should fail fast when the Arrow metadata does not match the contract."""
    from innovate import arrow_interchange

    table = pa.table({"value": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Arrow interchange metadata"):
        arrow_interchange.kernel_array_payload_from_table(table)
