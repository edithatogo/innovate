"""Experimental DataFrame engine helpers behind the Arrow contract."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import pandas as pd

from . import arrow_interchange
from .kernel import KERNEL_SCHEMA_VERSION, KernelJSONValue, KernelTablePayload

DataFrameEngine = Literal["pandas+pyarrow", "pandas", "polars"]
_PANDAS_ENGINES = {"pandas", "pandas+pyarrow"}


def _normalize_engine(engine: str) -> str:
    normalized = engine.strip().lower().replace("_", "-")
    if normalized in {"pandas", "pandas-pyarrow", "pandas+pyarrow"}:
        return "pandas+pyarrow"
    if normalized == "polars":
        return "polars"
    raise ValueError(f"Unsupported DataFrame engine: {engine}")


def _polars_module() -> Any:
    return importlib.import_module("polars")


def dataframe_engine_available(engine: str) -> bool:
    """Return whether an experimental DataFrame engine is available."""
    normalized = _normalize_engine(engine)
    if normalized in _PANDAS_ENGINES:
        return True
    return importlib.util.find_spec(normalized) is not None


@dataclass(frozen=True, slots=True)
class DataFrameEngineBenchmarkFixture:
    """Reproducible metadata for a tabular-engine benchmark comparison."""

    workload: str
    baseline_engine: str
    candidate_engines: tuple[str, ...]
    row_count: int
    column_count: int
    metrics: dict[str, KernelJSONValue] = field(default_factory=dict)
    attribution: dict[str, bool] = field(default_factory=dict)
    fallback: dict[str, KernelJSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, KernelJSONValue | list[str] | dict[str, KernelJSONValue] | dict[str, bool]]:
        """Serialize the benchmark fixture to a JSON-compatible dictionary."""
        return {
            "workload": self.workload,
            "baseline_engine": self.baseline_engine,
            "candidate_engines": list(self.candidate_engines),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "metrics": dict(self.metrics),
            "attribution": dict(self.attribution),
            "fallback": dict(self.fallback),
        }


def describe_dataframe_engine_experiments() -> dict[str, Any]:
    """Describe controlled DataFrame engine experimentation boundaries."""
    return {
        "schema_version": KERNEL_SCHEMA_VERSION,
        "default_surface": "pandas+pyarrow",
        "inventory": {
            "pandas": (
                "Python-facing tabular outputs",
                "preprocessing helpers",
                "model comparison utilities",
                "plot input ergonomics",
            ),
            "pyarrow": (
                "kernel array payload transport",
                "kernel table payload transport",
                "diagnostics artifact interchange",
                "binding-compatible table metadata",
            ),
            "polars": ("optional downstream Arrow consumer",),
        },
        "candidate_workloads": (
            "benchmark_corpus_metadata",
            "diagnostics_artifact_tables",
            "kernel_table_roundtrip",
            "model_card_refresh_tables",
        ),
        "metrics": (
            "row_count",
            "column_count",
            "correctness_hash",
            "wall_time_ms",
            "peak_memory_bytes",
        ),
        "optional_engines": {
            "polars": {
                "support_tier": "experimental",
                "dependency_extra": "dataframe",
                "fallback": "pandas+pyarrow",
            },
        },
        "public_contract": "kernel schema and Arrow-compatible payloads",
        "blocked_public_contracts": (
            "Polars lazy query plans",
            "engine-specific expression trees",
            "XLA compiler internals",
        ),
        "promotion_criteria": (
            "correctness parity with pandas+pyarrow",
            "reproducible benchmark evidence",
            "no public API drift",
            "explicit optional dependency gate",
        ),
        "attribution": {
            "tabular_execution": "DataFrame engine, query planning, and Arrow table conversion",
            "separate_from": "XLA-backed numerical kernels",
        },
    }


def _pandas_frame_from_payload(payload: KernelTablePayload) -> pd.DataFrame:
    frame = arrow_interchange.kernel_table_payload_to_dataframe(payload)
    frame = frame.convert_dtypes(dtype_backend="pyarrow")
    frame.attrs["innovate.kind"] = arrow_interchange.ARROW_INTERCHANGE_TABLE_KIND
    frame.attrs["innovate.schema_version"] = KERNEL_SCHEMA_VERSION
    frame.attrs["innovate.columns"] = list(payload.columns)
    frame.attrs["innovate.metadata"] = dict(payload.metadata)
    frame.attrs["innovate.dataframe_engine"] = "pandas+pyarrow"
    return frame


def _fallback_frame(payload: KernelTablePayload, *, requested_engine: str, reason: str) -> pd.DataFrame:
    frame = _pandas_frame_from_payload(payload)
    frame.attrs["innovate.requested_dataframe_engine"] = requested_engine
    frame.attrs["innovate.engine_fallback"] = reason
    return frame


def kernel_table_payload_to_experimental_dataframe(
    payload: KernelTablePayload,
    *,
    engine: str = "pandas+pyarrow",
    allow_fallback: bool = True,
) -> Any:
    """Convert a kernel table payload through an explicit DataFrame engine gate."""
    normalized = _normalize_engine(engine)
    if normalized in _PANDAS_ENGINES:
        return _pandas_frame_from_payload(payload)

    if not dataframe_engine_available(normalized):
        if allow_fallback:
            return _fallback_frame(payload, requested_engine=normalized, reason="missing_optional_dependency")
        raise ImportError(
            "polars is required for the experimental Polars DataFrame path; "
            "install innovate[dataframe] or allow fallback to pandas+pyarrow"
        )

    table = arrow_interchange.kernel_table_payload_to_table(payload)
    polars = _polars_module()
    return polars.from_arrow(table)


def kernel_table_payload_from_experimental_dataframe(
    frame: Any,
    *,
    metadata: Mapping[str, KernelJSONValue] | None = None,
) -> KernelTablePayload:
    """Convert an experimental DataFrame result back into a kernel table payload."""
    if isinstance(frame, pd.DataFrame):
        return arrow_interchange.kernel_table_payload_from_dataframe(frame, metadata=metadata)

    if frame.__class__.__module__.startswith("polars"):
        rows = tuple(
            tuple(cast(Mapping[str, KernelJSONValue], record)[column] for column in frame.columns)
            for record in frame.to_dicts()
        )
        return KernelTablePayload.from_rows(
            columns=tuple(str(column) for column in frame.columns), rows=rows, metadata=metadata
        )

    raise TypeError("Expected a pandas or experimental Polars DataFrame")


def _correctness_hash(payload: KernelTablePayload) -> str:
    serialized = json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_dataframe_engine_benchmark_fixture(
    payload: KernelTablePayload,
    *,
    workload: str = "kernel_table_roundtrip",
    candidate_engines: Sequence[str] = ("polars",),
) -> DataFrameEngineBenchmarkFixture:
    """Build reproducible benchmark metadata for tabular engine comparisons."""
    return DataFrameEngineBenchmarkFixture(
        workload=workload,
        baseline_engine="pandas+pyarrow",
        candidate_engines=tuple(candidate_engines),
        row_count=len(payload.rows),
        column_count=len(payload.columns),
        metrics={
            "correctness_hash": _correctness_hash(payload),
            "wall_time_ms": 0.0,
            "peak_memory_bytes": 0,
        },
        attribution={
            "tabular_execution": True,
            "xla_numerical_kernel": False,
        },
        fallback={
            "default": "pandas+pyarrow",
            "polars_missing": "pandas+pyarrow",
        },
    )


__all__ = [
    "DataFrameEngine",
    "DataFrameEngineBenchmarkFixture",
    "build_dataframe_engine_benchmark_fixture",
    "dataframe_engine_available",
    "describe_dataframe_engine_experiments",
    "kernel_table_payload_from_experimental_dataframe",
    "kernel_table_payload_to_experimental_dataframe",
]
