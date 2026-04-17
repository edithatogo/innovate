"""Tests for the benchmark runner contract."""

from __future__ import annotations

import json

from innovate.benchmarks import BenchmarkRunner, get_benchmark_case
from innovate.diffuse.bass import BassModel


def test_benchmark_runner_emits_standardized_metrics_and_diagnostics(tmp_path) -> None:
    """Benchmark execution should return comparable metrics and diagnostics."""
    case = get_benchmark_case("bass_smoke_adoption")
    runner = BenchmarkRunner()

    result = runner.run(BassModel(), case)

    assert result.case_id == case.case_id
    assert result.model_key == "bass"
    assert result.model_name == "BassModel"
    assert result.predictions.shape == case.observed.shape
    assert result.metrics["RMSE"] >= 0.0
    assert result.diagnostics.support_level in {"supported", "partial"}
    assert result.uncertainty.report_type == "point_estimate"

    payload = result.to_dict()
    assert payload["case_id"] == case.case_id
    assert payload["metrics"]["RMSE"] >= 0.0
    assert payload["diagnostics"]["support_level"] in {"supported", "partial"}

    output_path = tmp_path / "benchmark-result.json"
    result.write_json(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["model_key"] == "bass"
    assert saved["metrics"]["RMSE"] >= 0.0
    assert saved["diagnostics"]["support_level"] in {"supported", "partial"}

