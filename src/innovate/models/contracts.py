"""Canonical input contracts for network and policy diffusion workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from innovate.utils.validation import validate_sequence_numeric


def _as_2d_array(values: Sequence[Sequence[float]] | np.ndarray, param_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Parameter '{param_name}' must be a 2D array-like structure")
    return arr


@dataclass(frozen=True, slots=True)
class NetworkDiffusionInputs:
    """Serializable network structure used by network-aware diffusion models."""

    adjacency: np.ndarray
    node_labels: tuple[str, ...]
    coordinates: np.ndarray | None = None

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Sequence[Sequence[float]] | np.ndarray,
        node_labels: Sequence[str] | None = None,
        coordinates: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> NetworkDiffusionInputs:
        adjacency_arr = _as_2d_array(adjacency, "adjacency")
        if adjacency_arr.shape[0] != adjacency_arr.shape[1]:
            raise ValueError("Adjacency matrix must be square")

        if node_labels is None:
            labels = tuple(str(index) for index in range(adjacency_arr.shape[0]))
        else:
            labels = tuple(node_labels)
            if len(labels) != adjacency_arr.shape[0]:
                raise ValueError("Node labels must match adjacency dimensions")

        coordinate_arr = None
        if coordinates is not None:
            coordinate_arr = _as_2d_array(coordinates, "coordinates")
            if coordinate_arr.shape[0] != adjacency_arr.shape[0]:
                raise ValueError("Coordinates must match adjacency dimensions")

        np.fill_diagonal(adjacency_arr, 0.0)
        return cls(adjacency=adjacency_arr, node_labels=labels, coordinates=coordinate_arr)

    @classmethod
    def from_edge_list(
        cls,
        edge_list: Sequence[tuple[str, str] | tuple[str, str, float]],
        node_labels: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        directed: bool = False,
    ) -> NetworkDiffusionInputs:
        nodes = list(node_labels or [])
        if not nodes:
            seen: list[str] = []
            for edge in edge_list:
                for node in edge[:2]:
                    if node not in seen:
                        seen.append(node)
            nodes = seen

        index = {node: idx for idx, node in enumerate(nodes)}
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)

        if weights is not None and len(weights) != len(edge_list):
            raise ValueError("Weights must match the number of edges")

        for edge_index, edge in enumerate(edge_list):
            if len(edge) not in (2, 3):
                raise ValueError("Edge entries must contain two node labels and an optional weight")

            left, right = edge[:2]
            weight = float(weights[edge_index] if weights is not None else edge[2] if len(edge) == 3 else 1.0)
            adjacency[index[left], index[right]] = weight
            if not directed:
                adjacency[index[right], index[left]] = weight

        return cls.from_adjacency(adjacency, node_labels=nodes)

    @classmethod
    def from_coordinates(
        cls,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        node_labels: Sequence[str] | None = None,
    ) -> NetworkDiffusionInputs:
        coordinate_arr = _as_2d_array(coordinates, "coordinates")
        distances = np.linalg.norm(coordinate_arr[:, None, :] - coordinate_arr[None, :, :], axis=-1)
        with np.errstate(divide="ignore"):
            adjacency = np.where(distances > 0, 1.0 / (1.0 + distances), 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return cls.from_adjacency(adjacency, node_labels=node_labels, coordinates=coordinate_arr)

    def row_normalized_adjacency(self) -> np.ndarray:
        """Return a row-normalized copy of the adjacency matrix."""
        adjacency = np.asarray(self.adjacency, dtype=float)
        row_sums = adjacency.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(adjacency)
        np.divide(adjacency, row_sums, out=normalized, where=row_sums != 0)
        return normalized

    def to_dict(self) -> dict[str, object]:
        """Serialize the inputs to a JSON-friendly dictionary."""
        return {
            "adjacency": self.adjacency.tolist(),
            "node_labels": list(self.node_labels),
            "coordinates": None if self.coordinates is None else self.coordinates.tolist(),
        }


@dataclass(frozen=True, slots=True)
class PolicyTimingInputs:
    """Serializable timing and event-effect inputs for policy diffusion workflows."""

    event_times: tuple[float, ...]
    event_effects: tuple[float, ...]
    event_labels: tuple[str, ...] = ()

    @classmethod
    def from_events(
        cls,
        event_times: Sequence[float],
        event_effects: Sequence[float],
        event_labels: Sequence[str] | None = None,
    ) -> PolicyTimingInputs:
        times = validate_sequence_numeric(event_times, "event_times")
        effects = validate_sequence_numeric(event_effects, "event_effects")
        if len(times) != len(effects):
            raise ValueError("event_times and event_effects must have the same length")

        if event_labels is None:
            labels = tuple(f"event_{index}" for index in range(len(times)))
        else:
            labels = tuple(event_labels)
            if len(labels) != len(times):
                raise ValueError("event_labels must match event_times length")

        return cls(
            event_times=tuple(float(value) for value in times),
            event_effects=tuple(float(value) for value in effects),
            event_labels=labels,
        )

    def effect_profile(self, t: Sequence[float], decay: float) -> np.ndarray:
        """Compute the cumulative policy influence at each time point."""
        t_arr = validate_sequence_numeric(t, "t", allow_empty=True)
        if len(self.event_times) == 0:
            return np.zeros_like(t_arr)

        event_times = np.asarray(self.event_times, dtype=float)
        event_effects = np.asarray(self.event_effects, dtype=float)
        profile = np.zeros_like(t_arr, dtype=float)

        for event_time, event_effect in zip(event_times, event_effects, strict=True):
            active = t_arr >= event_time
            if not np.any(active):
                continue
            profile[active] += event_effect * np.exp(-decay * (t_arr[active] - event_time))
        return profile

    def to_dict(self) -> dict[str, object]:
        """Serialize the timing inputs to a JSON-friendly dictionary."""
        return {
            "event_times": list(self.event_times),
            "event_effects": list(self.event_effects),
            "event_labels": list(self.event_labels),
        }
