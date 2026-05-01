"""Fast benchmark corpus automation and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .corpus import BENCHMARK_METADATA_SCHEMA_VERSION, BenchmarkCase, list_benchmark_cases
from .model_cards import ModelCard, list_model_cards

ValidationSeverity = Literal["error", "warning"]

REQUIRED_BENCHMARK_METADATA_KEYS = (
    "scenario",
    "target",
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
)
VALID_RUNTIME_TIERS = ("fast_ci", "scheduled", "manual")
VALID_CI_POLICIES = ("fast", "workflow_dispatch", "scheduled")
VALID_COST_ESTIMATES = ("low", "medium", "high")


@dataclass(frozen=True, slots=True)
class BenchmarkValidationIssue:
    """Actionable benchmark metadata or model-card validation issue."""

    code: str
    message: str
    scope: str
    identifier: str
    severity: ValidationSeverity = "error"

    def to_dict(self) -> dict[str, str]:
        """Serialize the validation issue for automation logs."""
        return {
            "code": self.code,
            "message": self.message,
            "scope": self.scope,
            "identifier": self.identifier,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkAutomationReport:
    """Fast benchmark corpus validation report."""

    issues: tuple[BenchmarkValidationIssue, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return whether the report has no error-level issues."""
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for CI and scheduled jobs."""
        return {
            "ok": self.ok,
            "summary": dict(self.summary),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def assert_valid(self) -> None:
        """Raise a concise assertion error if validation failed."""
        if self.ok:
            return
        messages = "; ".join(f"{issue.identifier}: {issue.message}" for issue in self.issues)
        raise AssertionError(f"Benchmark corpus validation failed: {messages}")


def _validate_case(case: BenchmarkCase) -> list[BenchmarkValidationIssue]:
    issues: list[BenchmarkValidationIssue] = []
    metadata = case.metadata
    for key in REQUIRED_BENCHMARK_METADATA_KEYS:
        if key not in metadata or not str(metadata[key]).strip():
            issues.append(
                BenchmarkValidationIssue(
                    code="missing_metadata",
                    message=f"Benchmark case is missing required metadata key '{key}'.",
                    scope="benchmark_case",
                    identifier=case.case_id,
                ),
            )

    if metadata.get("metadata_schema_version") != BENCHMARK_METADATA_SCHEMA_VERSION:
        issues.append(
            BenchmarkValidationIssue(
                code="metadata_schema_mismatch",
                message=(f"Benchmark metadata schema version must be {BENCHMARK_METADATA_SCHEMA_VERSION}."),
                scope="benchmark_case",
                identifier=case.case_id,
            ),
        )
    if metadata.get("runtime_tier") not in VALID_RUNTIME_TIERS:
        issues.append(
            BenchmarkValidationIssue(
                code="invalid_runtime_tier",
                message=f"runtime_tier must be one of {', '.join(VALID_RUNTIME_TIERS)}.",
                scope="benchmark_case",
                identifier=case.case_id,
            ),
        )
    if metadata.get("ci_policy") not in VALID_CI_POLICIES:
        issues.append(
            BenchmarkValidationIssue(
                code="invalid_ci_policy",
                message=f"ci_policy must be one of {', '.join(VALID_CI_POLICIES)}.",
                scope="benchmark_case",
                identifier=case.case_id,
            ),
        )
    if metadata.get("cost_estimate") not in VALID_COST_ESTIMATES:
        issues.append(
            BenchmarkValidationIssue(
                code="invalid_cost_estimate",
                message=f"cost_estimate must be one of {', '.join(VALID_COST_ESTIMATES)}.",
                scope="benchmark_case",
                identifier=case.case_id,
            ),
        )
    if metadata.get("runtime_tier") != "fast_ci" and metadata.get("ci_policy") == "fast":
        issues.append(
            BenchmarkValidationIssue(
                code="expensive_case_in_fast_ci",
                message="Scheduled or manual benchmark cases must not use ci_policy='fast'.",
                scope="benchmark_case",
                identifier=case.case_id,
            ),
        )
    return issues


def _validate_model_card(card: ModelCard, case_ids: set[str]) -> list[BenchmarkValidationIssue]:
    issues: list[BenchmarkValidationIssue] = []
    for case_id in card.benchmark_case_ids:
        if case_id not in case_ids:
            issues.append(
                BenchmarkValidationIssue(
                    code="unknown_benchmark_case",
                    message=f"Model card references unknown benchmark case '{case_id}'.",
                    scope="model_card",
                    identifier=card.model_key,
                ),
            )
    if not card.supported_backends:
        issues.append(
            BenchmarkValidationIssue(
                code="missing_supported_backends",
                message="Model card must list supported backends for benchmark interpretation.",
                scope="model_card",
                identifier=card.model_key,
            ),
        )
    return issues


def validate_benchmark_corpus() -> BenchmarkAutomationReport:
    """Validate benchmark metadata and model-card freshness without running benchmarks."""
    cases = tuple(list_benchmark_cases())
    cards = list_model_cards()
    case_ids = {case.case_id for case in cases}
    issues: list[BenchmarkValidationIssue] = []
    for case in cases:
        issues.extend(_validate_case(case))
    for card in cards.values():
        issues.extend(_validate_model_card(card, case_ids))

    summary = {
        "metadata_schema_version": BENCHMARK_METADATA_SCHEMA_VERSION,
        "case_count": len(cases),
        "model_card_count": len(cards),
        "fast_ci_case_count": sum(case.metadata["ci_policy"] == "fast" for case in cases),
        "scheduled_case_count": sum(case.metadata["ci_policy"] == "scheduled" for case in cases),
        "manual_case_count": sum(case.metadata["ci_policy"] == "workflow_dispatch" for case in cases),
    }
    return BenchmarkAutomationReport(issues=tuple(issues), summary=summary)


def refresh_model_card_summaries() -> dict[str, dict[str, Any]]:
    """Build refreshable model-card summaries tied to benchmark case versions."""
    cases = {case.case_id: case for case in list_benchmark_cases()}
    summaries: dict[str, dict[str, Any]] = {}
    for model_key, card in list_model_cards().items():
        case_versions = {
            case_id: cases[case_id].dataset_version for case_id in card.benchmark_case_ids if case_id in cases
        }
        summaries[model_key] = {
            "model_key": card.model_key,
            "model_name": card.model_name,
            "family": card.family,
            "metadata_schema_version": BENCHMARK_METADATA_SCHEMA_VERSION,
            "benchmark_case_ids": list(card.benchmark_case_ids),
            "benchmark_case_versions": case_versions,
            "supported_backends": list(card.supported_backends),
            "freshness": {
                "status": "current" if len(case_versions) == len(card.benchmark_case_ids) else "stale",
                "rule": "all referenced benchmark cases must exist with dataset versions",
            },
        }
    return summaries


def describe_benchmark_automation() -> dict[str, Any]:
    """Describe fast and opt-in benchmark automation commands."""
    return {
        "metadata_schema_version": BENCHMARK_METADATA_SCHEMA_VERSION,
        "fast_ci_command": "uv run python -m pytest tests/unit/test_benchmark_automation.py",
        "scheduled_or_manual_command": "uv run pytest --benchmark-only --benchmark-json=benchmark.json",
        "fast_checks": (
            "metadata schema validation",
            "model-card freshness validation",
            "CI policy validation",
        ),
        "promotion_gates": (
            "reference NumPy/SciPy timing recorded",
            "XLA compilation cost reported separately",
            "XLA steady-state runtime reported separately",
            "Rust-native candidates compared against schema-compatible reference outputs",
        ),
    }


__all__ = [
    "BENCHMARK_METADATA_SCHEMA_VERSION",
    "REQUIRED_BENCHMARK_METADATA_KEYS",
    "BenchmarkAutomationReport",
    "BenchmarkValidationIssue",
    "describe_benchmark_automation",
    "refresh_model_card_summaries",
    "validate_benchmark_corpus",
]
