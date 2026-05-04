"""Tests for benchmark documentation synchronization."""

from __future__ import annotations

from pathlib import Path


def test_benchmark_docs_pages_are_present() -> None:
    """The benchmark documentation pages should exist alongside the code."""
    docs_root = Path("docs/source")

    assert (docs_root / "innovate.benchmarks.rst").is_file()
    assert (docs_root / "innovate.benchmarks.automation.rst").is_file()
    assert (docs_root / "innovate.benchmarks.corpus.rst").is_file()
    assert (docs_root / "innovate.benchmarks.model_cards.rst").is_file()
    assert (docs_root / "innovate.benchmarks.runner.rst").is_file()


def test_benchmark_workflow_tutorial_mentions_canonical_helpers() -> None:
    """The workflow tutorial should explain the stable benchmark entry points."""
    tutorial = Path("docs/source/tutorials/benchmark_workflows.rst").read_text()

    assert "validate_benchmark_corpus" in tutorial
    assert "refresh_model_card_summaries" in tutorial
    assert "run_stable_benchmark_suite" in tutorial
    assert "list_model_cards" in tutorial
    assert "machine-readable" in tutorial.lower()
    assert "model cards" in tutorial.lower()
    assert "XLA compilation cost" in tutorial
    assert "steady-state runtime" in tutorial


def test_docs_toctrees_include_benchmarks() -> None:
    """The docs index should surface the benchmark API and tutorial pages."""
    package_docs = Path("docs/source/innovate.rst").read_text()
    tutorials_docs = Path("docs/source/tutorials.rst").read_text()

    assert "innovate.benchmarks" in package_docs
    assert "tutorials/benchmark_workflows" in tutorials_docs


def test_rust_benchmark_ci_job_is_documented() -> None:
    """The CI workflow should validate the Rust benchmark harness compiles."""
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "rust-benchmarks" in workflow
    assert "cargo bench --bench native_kernel --no-run" in workflow
    assert "cargo check --example profile_memory_native_kernels" in workflow


def test_benchmark_docs_describe_fast_and_opt_in_automation() -> None:
    """Benchmark docs should distinguish fast checks from opt-in benchmark runs."""
    benchmark_docs = Path("docs/source/innovate.benchmarks.rst").read_text()

    assert "validate_benchmark_corpus" in benchmark_docs
    assert "workflow_dispatch" in benchmark_docs
    assert "pytest --benchmark-only" in benchmark_docs
