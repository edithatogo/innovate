"""Tests for controlled DataFrame engine experimentation."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from innovate import kernel
from innovate.dataframe_engines import (
    build_dataframe_engine_benchmark_fixture,
    dataframe_engine_available,
    describe_dataframe_engine_experiments,
    kernel_table_payload_from_experimental_dataframe,
    kernel_table_payload_to_experimental_dataframe,
)


def _table_payload() -> kernel.KernelTablePayload:
    return kernel.KernelTablePayload.from_rows(
        columns=("time", "adoption"),
        rows=((1.0, 10.0), (2.0, 22.0), (3.0, 35.0)),
        metadata={"source": "synthetic"},
    )


def test_dataframe_experiment_contract_keeps_pandas_pyarrow_as_default() -> None:
    """The experiment policy should preserve engine-neutral public contracts."""
    contract = describe_dataframe_engine_experiments()

    assert contract["default_surface"] == "pandas+pyarrow"
    assert contract["optional_engines"]["polars"]["support_tier"] == "experimental"
    assert "benchmark_corpus_metadata" in contract["candidate_workloads"]
    assert "kernel schema and Arrow-compatible payloads" in contract["public_contract"]
    assert "Polars lazy query plans" in contract["blocked_public_contracts"]
    assert "XLA-backed numerical kernels" in contract["attribution"]["separate_from"]


def test_pandas_pyarrow_path_round_trips_kernel_table_payload() -> None:
    """The default path should remain a pandas DataFrame with Arrow metadata."""
    payload = _table_payload()

    frame = kernel_table_payload_to_experimental_dataframe(payload)

    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == ["time", "adoption"]
    assert frame.attrs["innovate.dataframe_engine"] == "pandas+pyarrow"
    assert frame.attrs["innovate.schema_version"] == kernel.KERNEL_SCHEMA_VERSION
    assert kernel_table_payload_from_experimental_dataframe(frame) == payload


def test_polars_path_is_optional_and_falls_back_to_pandas_when_missing() -> None:
    """Missing optional engines should not break the default public surface."""
    payload = _table_payload()
    polars_installed = importlib.util.find_spec("polars") is not None

    frame = kernel_table_payload_to_experimental_dataframe(payload, engine="polars", allow_fallback=True)

    assert dataframe_engine_available("polars") is polars_installed
    if polars_installed:
        assert frame.__class__.__module__.startswith("polars")
    else:
        assert isinstance(frame, pd.DataFrame)
        assert frame.attrs["innovate.dataframe_engine"] == "pandas+pyarrow"
        assert frame.attrs["innovate.requested_dataframe_engine"] == "polars"
        assert frame.attrs["innovate.engine_fallback"] == "missing_optional_dependency"


def test_polars_path_can_fail_fast_when_fallback_is_disabled() -> None:
    """Strict optional-engine calls should raise when the engine is unavailable."""
    if importlib.util.find_spec("polars") is not None:
        pytest.skip("Strict missing-dependency behavior only applies without polars installed")

    with pytest.raises(ImportError, match="polars"):
        kernel_table_payload_to_experimental_dataframe(_table_payload(), engine="polars", allow_fallback=False)


def test_dataframe_benchmark_fixture_records_attribution_boundaries() -> None:
    """Benchmark fixtures should attribute tabular and XLA effects separately."""
    fixture = build_dataframe_engine_benchmark_fixture(_table_payload())
    payload = fixture.to_dict()

    assert payload["workload"] == "kernel_table_roundtrip"
    assert payload["row_count"] == 3
    assert payload["column_count"] == 2
    assert payload["baseline_engine"] == "pandas+pyarrow"
    assert "polars" in payload["candidate_engines"]
    assert payload["attribution"]["tabular_execution"] is True
    assert payload["attribution"]["xla_numerical_kernel"] is False
    assert "correctness_hash" in payload["metrics"]


def test_unsupported_engine_raises_value_error() -> None:
    """An explicit request for an unknown engine should fail fast with a ValueError."""
    payload = _table_payload()
    with pytest.raises(ValueError, match="Unsupported DataFrame engine: unsupported"):
        kernel_table_payload_to_experimental_dataframe(payload, engine="unsupported")
