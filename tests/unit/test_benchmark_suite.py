"""Tests for the stable benchmark suite helpers."""

from __future__ import annotations

import json

import innovate
from innovate.benchmarks import (
    BenchmarkRunner,
    get_model_card,
    list_benchmark_jobs,
    run_stable_benchmark_suite,
)


def test_stable_benchmark_jobs_cover_core_families() -> None:
    """The suite should expose canonical jobs for stable model families."""
    jobs = list_benchmark_jobs(model_keys=("bass", "fisher_pry", "multi_product"))

    assert [job.model_key for job in jobs] == ["bass", "fisher_pry", "multi_product"]
    assert [job.case_id for job in jobs] == [
        "bass_smoke_adoption",
        "fisher_pry_replacement_smoke",
        "lotka_volterra_competition_smoke",
    ]
    assert jobs[0].model_name == get_model_card("bass").model_name
    assert jobs[0].family == "diffusion"


def test_stable_benchmark_suite_is_serializable(tmp_path) -> None:
    """Stable benchmark suite outputs should be reproducible and machine readable."""
    suite = run_stable_benchmark_suite(
        runner=BenchmarkRunner(),
        model_keys=("bass", "fisher_pry", "multi_product"),
    )

    assert len(suite.runs) == 3
    assert suite.runs[0].case_id == "bass_smoke_adoption"
    assert suite.runs[0].metrics["RMSE"] >= 0.0
    assert suite.runs[0].diagnostics.support_level in {"supported", "partial"}

    payload = suite.to_dict()
    assert payload["run_count"] == 3
    assert payload["runs"][0]["case_id"] == "bass_smoke_adoption"

    output_path = tmp_path / "benchmark-suite.json"
    suite.write_json(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["run_count"] == 3
    assert saved["runs"][0]["case_id"] == "bass_smoke_adoption"


def test_root_api_exposes_suite_helpers() -> None:
    """The package root should export the canonical suite helpers."""
    assert innovate.BenchmarkRunner is BenchmarkRunner
    assert innovate.list_benchmark_jobs is list_benchmark_jobs
    assert innovate.run_stable_benchmark_suite is run_stable_benchmark_suite
