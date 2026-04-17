"""Benchmark corpus definitions for stable diffusion, substitution, and competition cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class BenchmarkFamily(str, Enum):
    """Families represented in the benchmark corpus."""

    DIFFUSION = "diffusion"
    SUBSTITUTION = "substitution"
    COMPETITION = "competition"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Immutable benchmark case description with reproducible synthetic observations."""

    case_id: str
    family: BenchmarkFamily
    canonical_model_key: str
    dataset_version: str
    description: str
    time: np.ndarray
    observed: np.ndarray
    source: str = "synthetic"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the benchmark case payload."""
        if not self.case_id:
            raise ValueError("Benchmark case_id must be non-empty.")
        if not self.dataset_version:
            raise ValueError("Benchmark dataset_version must be non-empty.")
        if not self.source:
            raise ValueError("Benchmark source must be non-empty.")

        time = np.asarray(self.time, dtype=float)
        observed = np.asarray(self.observed, dtype=float)
        if time.ndim != 1:
            raise ValueError("Benchmark time must be one-dimensional.")
        if observed.ndim != 1:
            raise ValueError("Benchmark observed data must be one-dimensional.")
        if len(time) != len(observed):
            raise ValueError("Benchmark time and observed arrays must have the same length.")
        if len(time) == 0:
            raise ValueError("Benchmark cases must contain at least one observation.")

        metadata = dict(self.metadata)
        metadata.setdefault("family", self.family.value)
        metadata.setdefault("source", self.source)
        metadata.setdefault("case_id", self.case_id)
        metadata.setdefault("dataset_version", self.dataset_version)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark case for downstream artifacts."""
        return {
            "case_id": self.case_id,
            "family": self.family.value,
            "canonical_model_key": self.canonical_model_key,
            "dataset_version": self.dataset_version,
            "description": self.description,
            "source": self.source,
            "time": self.time.tolist(),
            "observed": self.observed.tolist(),
            "metadata": dict(self.metadata),
        }


def _bass_smoke_case() -> BenchmarkCase:
    time = np.arange(0, 12, dtype=float)
    p, q, m = 0.03, 0.38, 1000.0
    exp_term = np.exp(-(p + q) * time)
    observed = m * (1 - exp_term) / (1 + (q / p) * exp_term)
    return BenchmarkCase(
        case_id="bass_smoke_adoption",
        family=BenchmarkFamily.DIFFUSION,
        canonical_model_key="bass",
        dataset_version="2026.04",
        description="Synthetic Bass diffusion adoption curve for deterministic smoke testing.",
        source="synthetic",
        time=time,
        observed=observed,
        metadata={
            "scenario": "bass_smoke",
            "target": "cumulative_adoption",
        },
    )


def _logistic_smoke_case() -> BenchmarkCase:
    time = np.arange(0, 12, dtype=float)
    observed = 1000.0 / (1.0 + np.exp(-0.8 * (time - 6.0)))
    return BenchmarkCase(
        case_id="logistic_growth_smoke",
        family=BenchmarkFamily.DIFFUSION,
        canonical_model_key="logistic",
        dataset_version="2026.04",
        description="Synthetic logistic diffusion curve used to compare smooth S-curves.",
        source="synthetic",
        time=time,
        observed=observed,
        metadata={
            "scenario": "logistic_smoke",
            "target": "cumulative_adoption",
        },
    )


def _fisher_pry_smoke_case() -> BenchmarkCase:
    time = np.arange(0, 12, dtype=float)
    observed = 1.0 / (1.0 + np.exp(-1.0 * (time - 6.0)))
    return BenchmarkCase(
        case_id="fisher_pry_replacement_smoke",
        family=BenchmarkFamily.SUBSTITUTION,
        canonical_model_key="fisher_pry",
        dataset_version="2026.04",
        description="Synthetic substitution curve that mirrors a replacement share transition.",
        source="synthetic",
        time=time,
        observed=observed,
        metadata={
            "scenario": "fisher_pry_smoke",
            "target": "market_share",
        },
    )


def _lotka_volterra_smoke_case() -> BenchmarkCase:
    time = np.arange(0, 12, dtype=float)
    observed = 1.0 / (1.0 + np.exp(-0.65 * (time - 5.0)))
    return BenchmarkCase(
        case_id="lotka_volterra_competition_smoke",
        family=BenchmarkFamily.COMPETITION,
        canonical_model_key="lotka_volterra",
        dataset_version="2026.04",
        description="Synthetic competition curve that captures a focal-share transition.",
        source="synthetic",
        time=time,
        observed=observed,
        metadata={
            "scenario": "lotka_volterra_smoke",
            "target": "focal_share",
        },
    )


_BENCHMARK_CASES: Mapping[str, BenchmarkCase] = MappingProxyType(
    {
        case.case_id: case
        for case in (
            _bass_smoke_case(),
            _logistic_smoke_case(),
            _fisher_pry_smoke_case(),
            _lotka_volterra_smoke_case(),
        )
    },
)


def list_benchmark_cases() -> list[BenchmarkCase]:
    """Return the benchmark corpus in stable identifier order."""
    return [case for _, case in sorted(_BENCHMARK_CASES.items(), key=lambda item: item[0])]


def get_benchmark_case(case_id: str) -> BenchmarkCase:
    """Return a benchmark case by stable identifier."""
    try:
        return _BENCHMARK_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark case: {case_id}") from exc
