"""Tests for benchmark documentation synchronization."""

from __future__ import annotations

import json
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
    assert "Validate Rust migration inventory" in workflow
    assert "python3 -m json.tool inst/discovery_manifest.json" in workflow
    assert "python3 -m json.tool ../../docs/source/_static/rust_core_migration_inventory.json" in workflow
    assert "cargo check --benches --examples" in workflow
    assert "cargo bench --bench native_kernel --no-run" in workflow
    assert "cargo check --example profile_memory_native_kernels" in workflow
    assert "cargo package --list" in workflow


def test_rust_migration_inventory_is_machine_readable() -> None:
    """The Rust migration inventory should remain present and JSON-decodable."""
    inventory_path = Path("docs/source/_static/rust_core_migration_inventory.json")

    assert inventory_path.is_file()
    inventory = json.loads(inventory_path.read_text())
    assert isinstance(inventory, dict)
    assert inventory["schema_version"] == 1
    assert set(inventory["owner_values"]) == {"rust_native", "python_bridge", "python_reference"}
    assert {entry["operation"] for entry in inventory["inventory"]} >= {
        "discover_models",
        "fit_model",
        "predict_model",
        "simulate_model",
        "summarize_model",
        "diagnose_model",
    }


def test_rust_profiling_surfaces_are_packaged() -> None:
    """The Rust crate package should keep benchmark and profiling entry points."""
    cargo = Path("bindings/rust/Cargo.toml").read_text()
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert '"benches/**"' in cargo
    assert '"examples/**"' in cargo
    assert '"inst/**"' in cargo
    assert '"scripts/**"' in cargo

    assert "benches/native_kernel.rs" in workflow
    assert "examples/profile_memory_native_kernels.rs" in workflow
    assert "inst/discovery_manifest.json" in workflow
    assert "scripts/profile_native_kernels.sh" in workflow
    assert "scripts/profile_memory_native_kernels.sh" in workflow


def test_benchmark_docs_describe_fast_and_opt_in_automation() -> None:
    """Benchmark docs should distinguish fast checks from opt-in benchmark runs."""
    benchmark_docs = Path("docs/source/innovate.benchmarks.rst").read_text()

    assert "validate_benchmark_corpus" in benchmark_docs
    assert "workflow_dispatch" in benchmark_docs
    assert "pytest --benchmark-only" in benchmark_docs
