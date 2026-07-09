"""Validated dataset contracts for diffusion, policy, and network modeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from innovate.data.provenance import DatasetProvenance
from innovate.models.contracts import NetworkDiffusionInputs, PolicyTimingInputs

DATASET_CONTRACT_SCHEMA_VERSION = "1.0"
DatasetKind = Literal[
    "adoption",
    "substitution",
    "competition",
    "policy_timing",
    "network_edges",
]


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


@dataclass(frozen=True, slots=True)
class AdoptionDataset:
    """Adoption curve with optional market denominator."""

    time: np.ndarray
    adoption: np.ndarray
    denominator: np.ndarray | None = None
    unit: str = "count"
    provenance: DatasetProvenance | None = None
    schema_version: str = DATASET_CONTRACT_SCHEMA_VERSION
    kind: DatasetKind = "adoption"

    def __post_init__(self) -> None:
        time = _as_1d_float(self.time, "time")
        adoption = _as_1d_float(self.adoption, "adoption")
        if time.shape != adoption.shape:
            raise ValueError("time and adoption must have the same length")
        if self.denominator is not None:
            denominator = _as_1d_float(self.denominator, "denominator")
            if denominator.shape != time.shape:
                raise ValueError("denominator must match time length")
            if np.any(denominator <= 0):
                raise ValueError("denominator values must be positive")
            object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "adoption", adoption)
        if not self.unit.strip():
            raise ValueError("unit must be non-empty")
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "time": self.time.tolist(),
            "adoption": self.adoption.tolist(),
            "denominator": None if self.denominator is None else self.denominator.tolist(),
            "unit": self.unit,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SubstitutionDataset:
    """Market-share substitution series (bounded shares over time)."""

    time: np.ndarray
    share: np.ndarray
    product_labels: tuple[str, ...] = ()
    unit: str = "share"
    provenance: DatasetProvenance | None = None
    schema_version: str = DATASET_CONTRACT_SCHEMA_VERSION
    kind: DatasetKind = "substitution"

    def __post_init__(self) -> None:
        time = _as_1d_float(self.time, "time")
        share = np.asarray(self.share, dtype=float)
        if share.ndim == 1:
            share = share.reshape(-1, 1)
        if share.ndim != 2:
            raise ValueError("share must be 1D or 2D")
        if share.shape[0] != time.shape[0]:
            raise ValueError("share rows must match time length")
        if not np.all(np.isfinite(share)):
            raise ValueError("share must contain only finite values")
        if np.any(share < 0.0) or np.any(share > 1.0):
            raise ValueError("share values must be in [0, 1]")
        labels = self.product_labels or tuple(f"product_{index}" for index in range(share.shape[1]))
        if len(labels) != share.shape[1]:
            raise ValueError("product_labels must match share columns")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "share", share)
        object.__setattr__(self, "product_labels", tuple(labels))
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "time": self.time.tolist(),
            "share": self.share.tolist(),
            "product_labels": list(self.product_labels),
            "unit": self.unit,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompetitionDataset:
    """Panel of competing products/units over time."""

    time: np.ndarray
    unit_id: tuple[str, ...]
    product_id: tuple[str, ...]
    value: np.ndarray
    unit: str = "count"
    provenance: DatasetProvenance | None = None
    schema_version: str = DATASET_CONTRACT_SCHEMA_VERSION
    kind: DatasetKind = "competition"

    def __post_init__(self) -> None:
        time = _as_1d_float(self.time, "time")
        value = _as_1d_float(self.value, "value")
        if not (len(self.unit_id) == len(self.product_id) == time.shape[0] == value.shape[0]):
            raise ValueError("time, unit_id, product_id, and value must have equal length")
        if any(not str(item).strip() for item in self.unit_id):
            raise ValueError("unit_id entries must be non-empty")
        if any(not str(item).strip() for item in self.product_id):
            raise ValueError("product_id entries must be non-empty")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit_id", tuple(str(item) for item in self.unit_id))
        object.__setattr__(self, "product_id", tuple(str(item) for item in self.product_id))
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "time": self.time.tolist(),
            "unit_id": list(self.unit_id),
            "product_id": list(self.product_id),
            "value": self.value.tolist(),
            "unit": self.unit,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PolicyTimingDataset:
    """Policy event timing with effects and labels."""

    event_times: np.ndarray
    event_effects: np.ndarray
    event_labels: tuple[str, ...] = ()
    unit: str = "effect"
    provenance: DatasetProvenance | None = None
    schema_version: str = DATASET_CONTRACT_SCHEMA_VERSION
    kind: DatasetKind = "policy_timing"

    def __post_init__(self) -> None:
        times = _as_1d_float(self.event_times, "event_times")
        effects = _as_1d_float(self.event_effects, "event_effects")
        if times.shape != effects.shape:
            raise ValueError("event_times and event_effects must have the same length")
        labels = self.event_labels or tuple(f"event_{index}" for index in range(times.shape[0]))
        if len(labels) != times.shape[0]:
            raise ValueError("event_labels must match event_times length")
        object.__setattr__(self, "event_times", times)
        object.__setattr__(self, "event_effects", effects)
        object.__setattr__(self, "event_labels", tuple(str(label) for label in labels))
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_policy_timing_inputs(self) -> PolicyTimingInputs:
        return PolicyTimingInputs.from_events(
            event_times=self.event_times.tolist(),
            event_effects=self.event_effects.tolist(),
            event_labels=self.event_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "event_times": self.event_times.tolist(),
            "event_effects": self.event_effects.tolist(),
            "event_labels": list(self.event_labels),
            "unit": self.unit,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NetworkEdgeDataset:
    """Network edge list for network diffusion workflows."""

    source: tuple[str, ...]
    target: tuple[str, ...]
    weight: np.ndarray
    directed: bool = False
    unit: str = "weight"
    provenance: DatasetProvenance | None = None
    schema_version: str = DATASET_CONTRACT_SCHEMA_VERSION
    kind: DatasetKind = "network_edges"

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("source and target must be non-empty")
        if len(self.source) != len(self.target):
            raise ValueError("source and target must have equal length")
        weight = _as_1d_float(self.weight, "weight")
        if weight.shape[0] != len(self.source):
            raise ValueError("weight must match edge count")
        if np.any(weight < 0.0):
            raise ValueError("weight values must be non-negative")
        object.__setattr__(self, "source", tuple(str(item) for item in self.source))
        object.__setattr__(self, "target", tuple(str(item) for item in self.target))
        object.__setattr__(self, "weight", weight)
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_network_inputs(self) -> NetworkDiffusionInputs:
        edges = [
            (left, right, float(weight))
            for left, right, weight in zip(self.source, self.target, self.weight, strict=True)
        ]
        return NetworkDiffusionInputs.from_edge_list(edges, directed=self.directed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "source": list(self.source),
            "target": list(self.target),
            "weight": self.weight.tolist(),
            "directed": self.directed,
            "unit": self.unit,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


DatasetContract = AdoptionDataset | SubstitutionDataset | CompetitionDataset | PolicyTimingDataset | NetworkEdgeDataset


def attach_provenance(dataset: DatasetContract, provenance: DatasetProvenance) -> DatasetContract:
    """Return a copy of the dataset with provenance attached."""
    data = dataset.to_dict()
    data["provenance"] = provenance.to_dict()
    return dataset_from_dict(data)


def dataset_from_dict(data: Mapping[str, Any]) -> DatasetContract:
    """Deserialize a dataset contract from a JSON-friendly mapping."""
    kind = str(data.get("kind", ""))
    provenance = None
    if data.get("provenance") is not None:
        provenance = DatasetProvenance.from_dict(data["provenance"])
    if kind == "adoption":
        return AdoptionDataset(
            time=data["time"],
            adoption=data["adoption"],
            denominator=data.get("denominator"),
            unit=str(data.get("unit", "count")),
            provenance=provenance,
            schema_version=str(data.get("schema_version", DATASET_CONTRACT_SCHEMA_VERSION)),
        )
    if kind == "substitution":
        return SubstitutionDataset(
            time=data["time"],
            share=data["share"],
            product_labels=tuple(data.get("product_labels", ())),
            unit=str(data.get("unit", "share")),
            provenance=provenance,
            schema_version=str(data.get("schema_version", DATASET_CONTRACT_SCHEMA_VERSION)),
        )
    if kind == "competition":
        return CompetitionDataset(
            time=data["time"],
            unit_id=tuple(data["unit_id"]),
            product_id=tuple(data["product_id"]),
            value=data["value"],
            unit=str(data.get("unit", "count")),
            provenance=provenance,
            schema_version=str(data.get("schema_version", DATASET_CONTRACT_SCHEMA_VERSION)),
        )
    if kind == "policy_timing":
        return PolicyTimingDataset(
            event_times=data["event_times"],
            event_effects=data["event_effects"],
            event_labels=tuple(data.get("event_labels", ())),
            unit=str(data.get("unit", "effect")),
            provenance=provenance,
            schema_version=str(data.get("schema_version", DATASET_CONTRACT_SCHEMA_VERSION)),
        )
    if kind == "network_edges":
        return NetworkEdgeDataset(
            source=tuple(data["source"]),
            target=tuple(data["target"]),
            weight=data["weight"],
            directed=bool(data.get("directed", False)),
            unit=str(data.get("unit", "weight")),
            provenance=provenance,
            schema_version=str(data.get("schema_version", DATASET_CONTRACT_SCHEMA_VERSION)),
        )
    raise ValueError(f"unsupported dataset kind: {kind}")
