"""Tests for the benchmark corpus registry."""

from __future__ import annotations

import numpy as np
import pytest

from innovate.benchmarks import BenchmarkCase, BenchmarkFamily, get_benchmark_case, list_benchmark_cases


def test_benchmark_corpus_exposes_stable_sorted_identifiers() -> None:
    """The corpus should expose stable, sorted benchmark identifiers."""
    cases = list_benchmark_cases()

    assert cases
    assert all(isinstance(case, BenchmarkCase) for case in cases)
    assert [case.case_id for case in cases] == sorted(case.case_id for case in cases)
    assert {case.family for case in cases} == {
        BenchmarkFamily.DIFFUSION,
        BenchmarkFamily.SUBSTITUTION,
        BenchmarkFamily.COMPETITION,
    }


def test_benchmark_case_metadata_is_complete_and_reproducible() -> None:
    """Each benchmark case should expose stable metadata and synthetic observations."""
    case = get_benchmark_case("bass_smoke_adoption")

    assert case.case_id == "bass_smoke_adoption"
    assert case.dataset_version == "2026.04"
    assert case.family is BenchmarkFamily.DIFFUSION
    assert case.canonical_model_key == "bass"
    assert case.source == "synthetic"
    assert case.description
    assert case.time.ndim == 1
    assert case.observed.ndim == 1
    assert len(case.time) == len(case.observed)
    assert np.all(np.diff(case.time) > 0)
    assert np.all(np.diff(case.observed) >= -1e-12)
    assert case.metadata["scenario"] == "bass_smoke"
    assert case.metadata["family"] == "diffusion"


def test_unknown_benchmark_case_raises_key_error() -> None:
    """Unknown benchmark identifiers should fail loudly."""
    with pytest.raises(KeyError, match="Unknown benchmark case"):
        get_benchmark_case("not_a_real_case")
