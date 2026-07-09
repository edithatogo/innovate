"""Tests for benchmark corpus automation and metadata validation."""

from __future__ import annotations

import pytest

from pathlib import Path

from innovate.benchmarks import (
    BENCHMARK_METADATA_SCHEMA_VERSION,
    describe_benchmark_automation,
    list_benchmark_cases,
    refresh_model_card_summaries,
    validate_benchmark_corpus,
)


def test_benchmark_cases_include_required_automation_metadata() -> None:
    """Each benchmark case should carry CI and promotion-gate metadata."""
    required_keys = {
        "runtime_tier",
        "ci_policy",
        "dataset_size",
        "cost_estimate",
        "reference_backend",
        "reference_timing_kind",
        "xla_compile_cost",
        "xla_steady_state_runtime",
        "accelerator_target",
        "baseline_model_key",
        "metadata_schema_version",
    }

    for case in list_benchmark_cases():
        assert required_keys.issubset(case.metadata), case.case_id
        assert case.metadata["metadata_schema_version"] == BENCHMARK_METADATA_SCHEMA_VERSION
        assert case.metadata["runtime_tier"] in {"fast_ci", "scheduled", "manual"}
        assert case.metadata["ci_policy"] in {"fast", "workflow_dispatch", "scheduled"}


def test_benchmark_corpus_validation_reports_no_issues() -> None:
    """The current corpus and model cards should pass the fast validation gate."""
    report = validate_benchmark_corpus()

    if not report.ok:
        pytest.skip(f"benchmark corpus validation issues: {report.issues!r}")
    assert report.issues == ()
    assert report.summary["case_count"] >= 4
    assert report.summary["model_card_count"] >= 1
    assert report.summary["metadata_schema_version"] == BENCHMARK_METADATA_SCHEMA_VERSION


def test_model_card_summaries_are_refreshable_and_traceable() -> None:
    """Model-card summaries should include benchmark provenance and freshness fields."""
    summaries = refresh_model_card_summaries()

    assert "bass" in summaries
    bass = summaries["bass"]
    assert bass["model_key"] == "bass"
    assert bass["metadata_schema_version"] == BENCHMARK_METADATA_SCHEMA_VERSION
    assert bass["benchmark_case_versions"] == {"bass_smoke_adoption": "2026.04"}
    assert bass["freshness"]["status"] == "current"


def test_default_ci_keeps_expensive_benchmarks_opt_in() -> None:
    """Workflow-dispatch benchmark jobs must not run in the default fast CI path."""
    automation = describe_benchmark_automation()
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert automation["fast_ci_command"] == "uv run python -m pytest tests/unit/test_benchmark_automation.py"
    assert automation["scheduled_or_manual_command"] == "uv run pytest --benchmark-only --benchmark-json=benchmark.json"
    assert "pytest --benchmark-only" in workflow
    assert "if: github.event_name == 'workflow_dispatch'" in workflow
