"""MARS surrogate benchmark-gate metadata and opt-in harness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

MARS_SURROGATE_GATE_SCHEMA_VERSION = "1.0"

DecisionOutcome = Literal["promote", "defer", "reject"]
EvidenceStatus = Literal["metadata_only", "benchmark_recorded"]
GainAttribution = Literal["surrogate", "xla", "interaction", "none", "unknown"]


@dataclass(frozen=True, slots=True)
class MarsSurrogateBenchmarkCandidate:
    """Benchmark-gate metadata for one MARS surrogate candidate workflow."""

    candidate_id: str
    scenario: str
    expected_output: str
    reference_backend: str
    eligible_xla_alternative: str
    correctness_tolerance: dict[str, float]
    promotion_thresholds: dict[str, float]
    runtime_tier: str = "manual"
    ci_policy: str = "workflow_dispatch"
    dependency_cost: str = "unknown_until_lockfile_evidence"
    evidence_status: EvidenceStatus = "metadata_only"
    decision_outcome: DecisionOutcome = "defer"
    gain_attribution: GainAttribution = "unknown"
    failure_modes: tuple[str, ...] = (
        "mars_import_or_fit_failure",
        "surrogate_extrapolation_instability",
        "reference_parity_failure",
        "xla_candidate_compile_cost_exceeds_gain",
    )
    benchmark_metrics: dict[str, float | None] = field(
        default_factory=lambda: {
            "reference_runtime_seconds": None,
            "mars_runtime_seconds": None,
            "xla_compile_seconds": None,
            "xla_steady_state_runtime_seconds": None,
            "surrogate_rmse_ratio": None,
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate metadata for documentation and opt-in artifacts."""
        return {
            "schema_version": MARS_SURROGATE_GATE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "scenario": self.scenario,
            "expected_output": self.expected_output,
            "reference_backend": self.reference_backend,
            "eligible_xla_alternative": self.eligible_xla_alternative,
            "correctness_tolerance": dict(self.correctness_tolerance),
            "promotion_thresholds": dict(self.promotion_thresholds),
            "runtime_tier": self.runtime_tier,
            "ci_policy": self.ci_policy,
            "dependency_cost": self.dependency_cost,
            "evidence_status": self.evidence_status,
            "decision_outcome": self.decision_outcome,
            "gain_attribution": self.gain_attribution,
            "failure_modes": list(self.failure_modes),
            "benchmark_metrics": dict(self.benchmark_metrics),
        }


@dataclass(frozen=True, slots=True)
class MarsSurrogateGateIssue:
    """Validation issue for MARS surrogate benchmark-gate metadata."""

    code: str
    message: str
    candidate_id: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the issue for JSON reports."""
        return {
            "code": self.code,
            "message": self.message,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True, slots=True)
class MarsSurrogateGateReport:
    """Validation report for the MARS surrogate benchmark gate."""

    issues: tuple[MarsSurrogateGateIssue, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return whether the gate metadata is valid."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation report."""
        return {
            "ok": self.ok,
            "summary": dict(self.summary),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def assert_valid(self) -> None:
        """Raise an assertion if the gate metadata is invalid."""
        if self.ok:
            return
        messages = "; ".join(f"{issue.candidate_id}: {issue.message}" for issue in self.issues)
        raise AssertionError(f"MARS surrogate benchmark gate validation failed: {messages}")


def _candidate_definitions() -> tuple[MarsSurrogateBenchmarkCandidate, ...]:
    common_tolerance = {
        "max_rmse_ratio": 1.05,
        "max_mean_absolute_percentage_error": 0.03,
    }
    common_thresholds = {
        "min_surrogate_speedup": 1.5,
        "max_dependency_install_seconds": 30.0,
        "max_xla_compile_to_steady_state_ratio": 5.0,
    }
    return (
        MarsSurrogateBenchmarkCandidate(
            candidate_id="mars_adoption_curve_surrogate",
            scenario="Approximate cumulative adoption curves over repeated policy scenario evaluations.",
            expected_output="adoption_curve",
            reference_backend="numpy_scipy",
            eligible_xla_alternative="jax_xla_surrogate_candidate",
            correctness_tolerance=common_tolerance,
            promotion_thresholds=common_thresholds,
        ),
        MarsSurrogateBenchmarkCandidate(
            candidate_id="mars_policy_scenario_response_surrogate",
            scenario="Approximate policy scenario-response surfaces for sensitivity workflows.",
            expected_output="scenario_response_surface",
            reference_backend="numpy_scipy",
            eligible_xla_alternative="jax_xla_surrogate_candidate",
            correctness_tolerance=common_tolerance,
            promotion_thresholds=common_thresholds,
        ),
    )


def list_mars_surrogate_benchmark_candidates() -> tuple[MarsSurrogateBenchmarkCandidate, ...]:
    """Return MARS surrogate benchmark-gate candidates in stable order."""
    return _candidate_definitions()


def validate_mars_surrogate_benchmark_gate() -> MarsSurrogateGateReport:
    """Validate MARS surrogate gate metadata without importing or running MARS."""
    issues: list[MarsSurrogateGateIssue] = []
    candidates = list_mars_surrogate_benchmark_candidates()
    for candidate in candidates:
        if candidate.runtime_tier != "manual" or candidate.ci_policy != "workflow_dispatch":
            issues.append(
                MarsSurrogateGateIssue(
                    code="mars_gate_must_be_opt_in",
                    message="MARS surrogate benchmarks must remain outside fast CI until evidence is recorded.",
                    candidate_id=candidate.candidate_id,
                ),
            )
        if candidate.reference_backend != "numpy_scipy":
            issues.append(
                MarsSurrogateGateIssue(
                    code="missing_reference_backend",
                    message="MARS candidates must compare against the NumPy/SciPy reference path.",
                    candidate_id=candidate.candidate_id,
                ),
            )
        if candidate.eligible_xla_alternative != "jax_xla_surrogate_candidate":
            issues.append(
                MarsSurrogateGateIssue(
                    code="missing_xla_alternative",
                    message="MARS candidates must name the eligible XLA-backed alternative.",
                    candidate_id=candidate.candidate_id,
                ),
            )
        if candidate.decision_outcome == "promote" and candidate.evidence_status != "benchmark_recorded":
            issues.append(
                MarsSurrogateGateIssue(
                    code="promotion_without_evidence",
                    message="MARS cannot be promoted without benchmark evidence.",
                    candidate_id=candidate.candidate_id,
                ),
            )
        if candidate.gain_attribution != "unknown" and candidate.evidence_status == "metadata_only":
            issues.append(
                MarsSurrogateGateIssue(
                    code="attribution_without_evidence",
                    message="Gain attribution requires recorded benchmark evidence.",
                    candidate_id=candidate.candidate_id,
                ),
            )

    decision_outcomes: dict[str, int] = {}
    for candidate in candidates:
        decision_outcomes[candidate.decision_outcome] = decision_outcomes.get(candidate.decision_outcome, 0) + 1

    return MarsSurrogateGateReport(
        issues=tuple(issues),
        summary={
            "schema_version": MARS_SURROGATE_GATE_SCHEMA_VERSION,
            "candidate_count": len(candidates),
            "decision_outcomes": decision_outcomes,
            "fast_ci_behavior": "metadata_validation_only",
        },
    )


def describe_mars_surrogate_benchmark_gate() -> dict[str, Any]:
    """Describe the current MARS surrogate promotion decision and commands."""
    return {
        "schema_version": MARS_SURROGATE_GATE_SCHEMA_VERSION,
        "mars_dependency_policy": "not_declared_until_promotion",
        "fast_ci_behavior": "metadata_validation_only",
        "decision_outcome": "defer",
        "reference_backend": "numpy_scipy",
        "eligible_xla_alternative": "jax_xla_surrogate_candidate",
        "opt_in_command": (
            "uv run python -m innovate.benchmarks.mars_surrogate --write-json "
            "benchmark-results/mars-surrogate-gate.json"
        ),
        "promotion_requirements": (
            "record NumPy/SciPy reference runtime and correctness",
            "record MARS surrogate runtime and dependency cost",
            "record XLA compile cost separately from XLA steady-state runtime",
            "attribute gains to surrogate, XLA, interaction, or neither",
            "keep mars out of package metadata until the decision changes from defer",
        ),
    }


def build_mars_surrogate_gate_artifact() -> dict[str, Any]:
    """Build an opt-in JSON artifact without importing MARS."""
    report = validate_mars_surrogate_benchmark_gate()
    return {
        "gate": describe_mars_surrogate_benchmark_gate(),
        "mars_available": find_spec("mars") is not None,
        "candidates": [candidate.to_dict() for candidate in list_mars_surrogate_benchmark_candidates()],
        "validation": report.to_dict(),
    }


def write_mars_surrogate_gate_artifact(path: str | Path) -> Path:
    """Write the MARS surrogate gate artifact to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_mars_surrogate_gate_artifact(), indent=2, sort_keys=True))
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the opt-in MARS surrogate benchmark gate dry run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", type=Path, help="Write the gate artifact to this JSON path.")
    args = parser.parse_args(argv)

    artifact = build_mars_surrogate_gate_artifact()
    if args.write_json:
        write_mars_surrogate_gate_artifact(args.write_json)
    else:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through opt-in CLI.
    raise SystemExit(main())
