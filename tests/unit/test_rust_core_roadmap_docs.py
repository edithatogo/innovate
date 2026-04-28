"""Tests for the Rust core roadmap and binding-governance documentation."""

from __future__ import annotations

from pathlib import Path


def test_rust_core_roadmap_documentation_is_present() -> None:
    """The Rust core trajectory should be documented as a first-class roadmap."""
    docs_root = Path("docs/source")

    assert (docs_root / "rust_core_roadmap.rst").is_file()


def test_rust_core_roadmap_names_candidate_operations_and_gates() -> None:
    """The roadmap should make migration and promotion criteria explicit."""
    roadmap = Path("docs/source/rust_core_roadmap.rst").read_text()

    for operation in ("discover_models", "predict_model", "simulate_model", "fit_model", "summarize_model", "diagnose_model"):
        assert operation in roadmap

    assert "Python reference semantics" in roadmap
    assert "parity tests" in roadmap
    assert "benchmark gates" in roadmap
    assert "schema compatibility" in roadmap
    assert "Rust-native" in roadmap
    assert "logistic prediction" in roadmap
    assert "logistic fitting" in roadmap
    assert "logistic summary and diagnostics" in roadmap
    assert "Python bridge fallback" in roadmap
    assert "same logistic-native slice" in roadmap


def test_architecture_docs_surface_rust_core_strategy() -> None:
    """Architecture indices should link the Rust core strategy and ADR."""
    architecture = Path("docs/architecture_modernization_roadmap.md").read_text()
    principles = Path("docs/architecture_principles.md").read_text()
    index = Path("docs/source/index.rst").read_text()

    assert "Rust Core Runtime" in architecture
    assert "ADR 0004" in architecture
    assert "Rust Core Trajectory" in principles
    assert "rust_core_roadmap" in index
