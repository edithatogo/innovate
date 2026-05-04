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

    for operation in (
        "discover_models",
        "predict_model",
        "simulate_model",
        "fit_model",
        "summarize_model",
        "diagnose_model",
    ):
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
    assert "Benchmarking and profiling tooling" in roadmap
    assert "criterion" in roadmap
    assert "cargo-flamegraph" in roadmap
    assert "bindings/rust/benches/native_kernel.rs" in roadmap
    assert "bindings/rust/scripts/profile_native_kernels.sh" in roadmap
    assert "bindings/rust/examples/profile_memory_native_kernels.rs" in roadmap
    assert "bindings/rust/scripts/profile_memory_native_kernels.sh" in roadmap
    assert "DHAT" in roadmap
    assert "not entirely written in Rust yet" in roadmap
    assert "GPU profiling" in roadmap


def test_rust_core_roadmap_inventories_runtime_status_and_xla_fit() -> None:
    """The roadmap should inventory native, fallback, and Python-only status."""
    roadmap = Path("docs/source/rust_core_roadmap.rst").read_text()

    for phrase in (
        "Operation support inventory",
        "Native Rust scope",
        "Bridge fallback scope",
        "Python-only reference scope",
        "Rust vs JAX/XLA eligibility",
        "unsupported_native_operation",
        "bridge_command_failed",
        "XLA compile cost",
        "steady-state runtime",
        "promotion decision",
    ):
        assert phrase in roadmap

    expected_inventory = {
        "discover_models": ("Native metadata discovery", "Low"),
        "fit_model": ("Native logistic fitting", "Medium"),
        "predict_model": ("Native logistic prediction", "High"),
        "simulate_model": ("Native logistic simulation", "High"),
        "summarize_model": ("Native logistic summary", "Medium"),
        "diagnose_model": ("Native logistic diagnostics", "Medium"),
    }

    for operation, expected_phrases in expected_inventory.items():
        assert operation in roadmap
        for phrase in expected_phrases:
            assert phrase in roadmap


def test_rust_core_expansion_track_records_phase_one_inventory() -> None:
    """The archived track should keep the inventory decision visible."""
    track_dir = Path("conductor/archive/rust_core_expansion_20260430")
    spec = (track_dir / "spec.md").read_text()
    plan = (track_dir / "plan.md").read_text()
    track_text = spec + plan

    for phrase in (
        "operation support inventory",
        "native Rust scope",
        "bridge fallback scope",
        "Python-only reference scope",
        "Rust vs JAX/XLA eligibility",
        "benchmark promotion dossier",
    ):
        assert phrase in track_text


def test_architecture_docs_surface_rust_core_strategy() -> None:
    """Architecture indices should link the Rust core strategy and ADR."""
    architecture = Path("docs/architecture_modernization_roadmap.md").read_text()
    principles = Path("docs/architecture_principles.md").read_text()
    index = Path("docs/source/index.rst").read_text()

    assert "Rust Core Runtime" in architecture
    assert "ADR 0004" in architecture
    assert "Rust Core Trajectory" in principles
    assert "rust_core_roadmap" in index


def test_rust_binding_docs_surface_tracing_observability() -> None:
    """Rust binding docs should mention structured tracing on fallback paths."""
    docs = Path("docs/source/innovate.rust_bindings.rst").read_text()

    assert "tracing" in docs
    assert "native paths fall back" in docs
    assert "Python bridge fails" in docs
