"""Benchmark corpus and model-card utilities for innovate."""

from .automation import (
    BENCHMARK_METADATA_SCHEMA_VERSION,
    BenchmarkAutomationReport,
    BenchmarkValidationIssue,
    describe_benchmark_automation,
    refresh_model_card_summaries,
    validate_benchmark_corpus,
)
from .corpus import BenchmarkCase, BenchmarkFamily, get_benchmark_case, list_benchmark_cases
from .model_cards import ModelCard, get_model_card, list_model_cards
from .runner import (
    BenchmarkJob,
    BenchmarkRun,
    BenchmarkRunner,
    BenchmarkSuiteResult,
    list_benchmark_jobs,
    run_stable_benchmark_suite,
)

__all__ = [
    "BENCHMARK_METADATA_SCHEMA_VERSION",
    "MARS_SURROGATE_GATE_SCHEMA_VERSION",
    "BenchmarkAutomationReport",
    "BenchmarkCase",
    "BenchmarkFamily",
    "BenchmarkJob",
    "BenchmarkRun",
    "BenchmarkRunner",
    "BenchmarkSuiteResult",
    "BenchmarkValidationIssue",
    "MarsSurrogateBenchmarkCandidate",
    "MarsSurrogateGateIssue",
    "MarsSurrogateGateReport",
    "ModelCard",
    "build_mars_surrogate_gate_artifact",
    "describe_benchmark_automation",
    "describe_mars_surrogate_benchmark_gate",
    "get_benchmark_case",
    "get_model_card",
    "list_benchmark_cases",
    "list_benchmark_jobs",
    "list_mars_surrogate_benchmark_candidates",
    "list_model_cards",
    "refresh_model_card_summaries",
    "run_stable_benchmark_suite",
    "validate_benchmark_corpus",
    "validate_mars_surrogate_benchmark_gate",
    "write_mars_surrogate_gate_artifact",
]

_MARS_SURROGATE_EXPORTS = {
    "MARS_SURROGATE_GATE_SCHEMA_VERSION",
    "MarsSurrogateBenchmarkCandidate",
    "MarsSurrogateGateIssue",
    "MarsSurrogateGateReport",
    "build_mars_surrogate_gate_artifact",
    "describe_mars_surrogate_benchmark_gate",
    "list_mars_surrogate_benchmark_candidates",
    "validate_mars_surrogate_benchmark_gate",
    "write_mars_surrogate_gate_artifact",
}


def __getattr__(name: str) -> object:
    """Lazily expose MARS gate helpers without preloading the CLI module."""
    if name in _MARS_SURROGATE_EXPORTS:
        from . import mars_surrogate

        value = getattr(mars_surrogate, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
