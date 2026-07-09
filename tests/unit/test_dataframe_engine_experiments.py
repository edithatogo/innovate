"""Tests for controlled DataFrame engine experimentation."""

from __future__ import annotations

import importlib.util
from unittest import mock

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


def test_kernel_table_payload_from_experimental_dataframe_pandas() -> None:
    """It should extract a payload from a pandas DataFrame."""
    df = pd.DataFrame({"time": [1.0, 2.0], "adoption": [10.0, 20.0]})
    metadata = {"source": "test"}
    payload = kernel_table_payload_from_experimental_dataframe(df, metadata=metadata)
    assert payload.metadata == metadata
    assert payload.columns == ("time", "adoption")
    assert payload.rows == ((1.0, 10.0), (2.0, 20.0))


def test_kernel_table_payload_from_experimental_dataframe_polars() -> None:
    """It should extract a payload from a polars DataFrame."""
    if importlib.util.find_spec("polars") is None:
        pytest.skip("polars is not installed")

    import polars as pl

    df = pl.DataFrame({"time": [1.0, 2.0], "adoption": [10.0, 20.0]})
    metadata = {"source": "test"}
    payload = kernel_table_payload_from_experimental_dataframe(df, metadata=metadata)
    assert payload.metadata == metadata
    assert payload.columns == ("time", "adoption")
    assert payload.rows == ((1.0, 10.0), (2.0, 20.0))


def test_kernel_table_payload_from_experimental_dataframe_type_error() -> None:
    """It should raise TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Expected a pandas or experimental Polars DataFrame"):
        kernel_table_payload_from_experimental_dataframe([{"time": 1.0}])


def test_describe_dataframe_engine_experiments_comprehensive() -> None:
    """The describe_dataframe_engine_experiments function should return the complete contract."""
    contract = describe_dataframe_engine_experiments()

    assert contract["schema_version"] == kernel.KERNEL_SCHEMA_VERSION
    assert contract["default_surface"] == "pandas+pyarrow"

    assert "pandas" in contract["inventory"]
    assert "pyarrow" in contract["inventory"]
    assert "polars" in contract["inventory"]

    assert "Python-facing tabular outputs" in contract["inventory"]["pandas"]
    assert "optional downstream Arrow consumer" in contract["inventory"]["polars"]

    assert "benchmark_corpus_metadata" in contract["candidate_workloads"]
    assert "diagnostics_artifact_tables" in contract["candidate_workloads"]

    assert "row_count" in contract["metrics"]
    assert "column_count" in contract["metrics"]
    assert "correctness_hash" in contract["metrics"]
    assert "wall_time_ms" in contract["metrics"]
    assert "peak_memory_bytes" in contract["metrics"]

    assert contract["optional_engines"]["polars"]["support_tier"] == "experimental"
    assert contract["optional_engines"]["polars"]["dependency_extra"] == "dataframe"
    assert contract["optional_engines"]["polars"]["fallback"] == "pandas+pyarrow"

    assert contract["public_contract"] == "kernel schema and Arrow-compatible payloads"

    assert "Polars lazy query plans" in contract["blocked_public_contracts"]
    assert "engine-specific expression trees" in contract["blocked_public_contracts"]
    assert "XLA compiler internals" in contract["blocked_public_contracts"]

    assert "correctness parity with pandas+pyarrow" in contract["promotion_criteria"]
    assert "reproducible benchmark evidence" in contract["promotion_criteria"]
    assert "no public API drift" in contract["promotion_criteria"]
    assert "explicit optional dependency gate" in contract["promotion_criteria"]

    assert (
        contract["attribution"]["tabular_execution"] == "DataFrame engine, query planning, and Arrow table conversion"
    )
    assert contract["attribution"]["separate_from"] == "XLA-backed numerical kernels"


def test_unsupported_engine_raises_value_error() -> None:
    """An explicit request for an unknown engine should fail fast with a ValueError."""
    payload = _table_payload()
    with pytest.raises(ValueError, match="Unsupported DataFrame engine: unsupported"):
        kernel_table_payload_to_experimental_dataframe(payload, engine="unsupported")


def test_dataframe_engine_available_returns_true_for_pandas_engines() -> None:
    """Pandas variants should always report as available without import checks."""
    assert dataframe_engine_available("pandas") is True
    assert dataframe_engine_available("pandas+pyarrow") is True
    assert dataframe_engine_available("pandas-pyarrow") is True
    assert dataframe_engine_available("PANDAS") is True


def test_dataframe_engine_available_checks_importlib_for_polars() -> None:
    """Polars availability should depend on whether the module can be found."""
    with mock.patch("importlib.util.find_spec", return_value=mock.Mock()) as find_spec_mock:
        assert dataframe_engine_available("polars") is True
        find_spec_mock.assert_called_once_with("polars")

    with mock.patch("importlib.util.find_spec", return_value=None) as find_spec_mock:
        assert dataframe_engine_available("polars") is False
        find_spec_mock.assert_called_once_with("polars")


def test_dataframe_engine_available_raises_on_unsupported_engine() -> None:
    """Unknown engines should raise ValueError instead of returning False."""
    with pytest.raises(ValueError, match="Unsupported DataFrame engine: duckdb"):
        dataframe_engine_available("duckdb")
