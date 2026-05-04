"""Tests for MARS surrogate benchmark-gate metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from innovate.benchmarks import (
    MARS_SURROGATE_GATE_SCHEMA_VERSION,
    describe_mars_surrogate_benchmark_gate,
    list_mars_surrogate_benchmark_candidates,
    validate_mars_surrogate_benchmark_gate,
)


def test_mars_is_not_a_base_or_optional_dependency_before_promotion() -> None:
    """The benchmark gate must not promote mars into packaging metadata yet."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dependency_groups = pyproject.get("dependency-groups", {})
    project = pyproject["project"]
    dependency_sections = [
        project.get("dependencies", []),
        *project.get("optional-dependencies", {}).values(),
        *dependency_groups.values(),
    ]

    for dependencies in dependency_sections:
        assert not any(str(dependency).split(">=", maxsplit=1)[0] == "mars" for dependency in dependencies)


def test_mars_surrogate_candidates_compare_reference_and_xla_paths() -> None:
    """Candidate metadata should separate surrogate gains from XLA-backed gains."""
    candidates = list_mars_surrogate_benchmark_candidates()

    assert {candidate.candidate_id for candidate in candidates} == {
        "mars_adoption_curve_surrogate",
        "mars_policy_scenario_response_surrogate",
    }

    for candidate in candidates:
        payload = candidate.to_dict()

        assert payload["schema_version"] == MARS_SURROGATE_GATE_SCHEMA_VERSION
        assert payload["reference_backend"] == "numpy_scipy"
        assert payload["eligible_xla_alternative"] == "jax_xla_surrogate_candidate"
        assert payload["runtime_tier"] == "manual"
        assert payload["ci_policy"] == "workflow_dispatch"
        assert payload["decision_outcome"] == "defer"
        assert payload["evidence_status"] == "metadata_only"
        assert payload["gain_attribution"] == "unknown"
        assert payload["correctness_tolerance"]["max_rmse_ratio"] <= 1.05
        assert payload["promotion_thresholds"]["min_surrogate_speedup"] >= 1.25
        assert "mars_import_or_fit_failure" in payload["failure_modes"]


def test_mars_surrogate_gate_validation_records_no_fast_ci_execution() -> None:
    """The fast metadata gate should pass without importing or running mars."""
    report = validate_mars_surrogate_benchmark_gate()
    gate = describe_mars_surrogate_benchmark_gate()

    assert report.ok
    assert report.issues == ()
    assert report.summary["schema_version"] == MARS_SURROGATE_GATE_SCHEMA_VERSION
    assert report.summary["candidate_count"] == 2
    assert report.summary["decision_outcomes"] == {"defer": 2}
    assert gate["mars_dependency_policy"] == "not_declared_until_promotion"
    assert gate["fast_ci_behavior"] == "metadata_validation_only"
    assert gate["opt_in_command"] == (
        "uv run python -m innovate.benchmarks.mars_surrogate --write-json benchmark-results/mars-surrogate-gate.json"
    )


def test_mars_surrogate_docs_record_deferred_decision_and_thresholds() -> None:
    """Documentation should state the current decision and benchmark thresholds."""
    docs = Path("docs/source/innovate.benchmarks.mars_surrogate.rst").read_text()
    tutorial = Path("docs/source/tutorials/benchmark_workflows.rst").read_text()
    ecosystem = Path("docs/ecosystem/module_incubation_strategy.md").read_text()

    assert "defer" in docs.lower()
    assert "min_surrogate_speedup" in docs
    assert "jax_xla_surrogate_candidate" in docs
    assert "metadata_validation_only" in docs
    assert "MARS surrogate benchmark gate" in tutorial
    assert "MARS surrogate benchmark gate" in ecosystem
