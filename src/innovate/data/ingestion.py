"""Local dataset ingestion helpers (CSV, Parquet/Arrow, Polars, pandas)."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from innovate.data.contracts import (
    AdoptionDataset,
    CompetitionDataset,
    DatasetContract,
    DatasetKind,
    NetworkEdgeDataset,
    PolicyTimingDataset,
    SubstitutionDataset,
    attach_provenance,
)
from innovate.data.provenance import DatasetProvenance, compute_payload_checksum
from innovate.data.validation import ValidationReport, require_valid, validate_dataset

FrameLike = Any
SUPPORTED_LOCAL_FORMATS = ("csv", "parquet", "arrow", "pandas", "polars")


def polars_available() -> bool:
    """Return whether the optional Polars dependency is importable."""
    return importlib.util.find_spec("polars") is not None


def _to_pandas(frame: FrameLike) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    module_name = type(frame).__module__
    if module_name.startswith("polars"):
        if not hasattr(frame, "to_pandas"):
            raise TypeError("Polars frame must provide to_pandas()")
        return frame.to_pandas()
    raise TypeError(f"unsupported frame type: {type(frame)!r}")


def load_table(path: str | Path, *, format: str | None = None) -> pd.DataFrame:
    """Load a local table as pandas DataFrame from CSV, Parquet, or Arrow IPC."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    fmt = (format or path.suffix.lstrip(".")).lower()
    if fmt in {"csv", "txt"}:
        return pd.read_csv(path)
    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if fmt in {"arrow", "feather", "ipc"}:
        from pyarrow import feather

        return feather.read_feather(path)
    raise ValueError(f"unsupported local format '{fmt}'; supported={list(SUPPORTED_LOCAL_FORMATS)}")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def frame_to_dataset(
    frame: FrameLike,
    kind: DatasetKind,
    *,
    provenance: DatasetProvenance | None = None,
    unit: str | None = None,
    directed: bool = False,
    require_provenance: bool = True,
    validate: bool = True,
) -> DatasetContract:
    """Convert a tabular frame into a validated dataset contract."""
    table = _to_pandas(frame)
    if require_provenance and provenance is None:
        raise ValueError("provenance is required (fail closed)")

    dataset: DatasetContract
    if kind == "adoption":
        _require_columns(table, ("time", "adoption"))
        dataset = AdoptionDataset(
            time=table["time"].to_numpy(dtype=float),
            adoption=table["adoption"].to_numpy(dtype=float),
            denominator=table["denominator"].to_numpy(dtype=float) if "denominator" in table.columns else None,
            unit=unit or ("share" if "denominator" in table.columns else "count"),
            provenance=provenance,
        )
    elif kind == "substitution":
        _require_columns(table, ("time",))
        share_cols = [column for column in table.columns if column.startswith("share")]
        if not share_cols and "share" in table.columns:
            share_cols = ["share"]
        if not share_cols:
            raise ValueError("substitution frames require 'share' or share_* columns")
        labels = tuple(
            column.removeprefix("share_").removeprefix("share") or f"product_{index}"
            for index, column in enumerate(share_cols)
        )
        labels = tuple(
            f"product_{index}" if label in {"", "_"} else label.lstrip("_") for index, label in enumerate(labels)
        )
        dataset = SubstitutionDataset(
            time=table["time"].to_numpy(dtype=float),
            share=table[list(share_cols)].to_numpy(dtype=float),
            product_labels=labels,
            unit=unit or "share",
            provenance=provenance,
        )
    elif kind == "competition":
        _require_columns(table, ("time", "unit_id", "product_id", "value"))
        dataset = CompetitionDataset(
            time=table["time"].to_numpy(dtype=float),
            unit_id=tuple(table["unit_id"].astype(str).tolist()),
            product_id=tuple(table["product_id"].astype(str).tolist()),
            value=table["value"].to_numpy(dtype=float),
            unit=unit or "count",
            provenance=provenance,
        )
    elif kind == "policy_timing":
        _require_columns(table, ("event_times", "event_effects"))
        labels = tuple(table["event_labels"].astype(str).tolist()) if "event_labels" in table.columns else ()
        dataset = PolicyTimingDataset(
            event_times=table["event_times"].to_numpy(dtype=float),
            event_effects=table["event_effects"].to_numpy(dtype=float),
            event_labels=labels,
            unit=unit or "effect",
            provenance=provenance,
        )
    elif kind == "network_edges":
        _require_columns(table, ("source", "target"))
        weight = (
            table["weight"].to_numpy(dtype=float) if "weight" in table.columns else np.ones(len(table), dtype=float)
        )
        dataset = NetworkEdgeDataset(
            source=tuple(table["source"].astype(str).tolist()),
            target=tuple(table["target"].astype(str).tolist()),
            weight=weight,
            directed=directed,
            unit=unit or "weight",
            provenance=provenance,
        )
    else:
        raise ValueError(f"unsupported dataset kind: {kind}")

    if provenance is not None and dataset.provenance is not None and not dataset.provenance.checksum:
        checksum = compute_payload_checksum(dataset.to_dict())
        dataset = attach_provenance(dataset, provenance.with_checksum(checksum))

    if validate:
        require_valid(dataset)
    return dataset


def ingest_local(
    path: str | Path,
    kind: DatasetKind,
    *,
    provenance: DatasetProvenance,
    format: str | None = None,
    unit: str | None = None,
    directed: bool = False,
    validate: bool = True,
) -> tuple[DatasetContract, ValidationReport]:
    """Ingest a local file into a dataset contract with validation report."""
    steps = [*provenance.transform_steps, f"load:{Path(path).name}"]
    provenance = DatasetProvenance(
        source=provenance.source,
        license=provenance.license,
        extraction_time=provenance.extraction_time,
        transform_steps=tuple(steps),
        schema_version=provenance.schema_version,
        checksum=provenance.checksum,
        citation=provenance.citation,
        extra={**dict(provenance.extra), "path": str(path), "format": format or Path(path).suffix},
    )
    frame = load_table(path, format=format)
    dataset = frame_to_dataset(
        frame,
        kind,
        provenance=provenance,
        unit=unit,
        directed=directed,
        require_provenance=True,
        validate=validate,
    )
    report = validate_dataset(dataset)
    return dataset, report


def ingest_polars(
    frame: FrameLike,
    kind: DatasetKind,
    *,
    provenance: DatasetProvenance,
    unit: str | None = None,
    directed: bool = False,
    validate: bool = True,
) -> tuple[DatasetContract, ValidationReport]:
    """Ingest a Polars DataFrame when the optional dependency is installed."""
    if not polars_available() and type(frame).__module__.startswith("polars"):
        raise ImportError("polars is not installed; install innovate[polars] or pass a pandas frame")
    steps = [*provenance.transform_steps, "frame:polars"]
    provenance = DatasetProvenance(
        source=provenance.source,
        license=provenance.license,
        extraction_time=provenance.extraction_time,
        transform_steps=tuple(steps),
        schema_version=provenance.schema_version,
        checksum=provenance.checksum,
        citation=provenance.citation,
        extra={**dict(provenance.extra), "frame": "polars"},
    )
    dataset = frame_to_dataset(
        frame,
        kind,
        provenance=provenance,
        unit=unit,
        directed=directed,
        validate=validate,
    )
    return dataset, validate_dataset(dataset)


def reproducible_artifact(dataset: DatasetContract) -> dict[str, Any]:
    """Serialize dataset + provenance into a stable JSON-friendly artifact."""
    payload = dataset.to_dict()
    if dataset.provenance is None:
        raise ValueError("dataset provenance is required for reproducible artifacts")
    return {
        "artifact_kind": "innovate.dataset",
        "dataset": payload,
        "formats_supported": list(SUPPORTED_LOCAL_FORMATS),
        "polars_available": polars_available(),
    }
