"""Tests for the Rust core roadmap and binding-governance documentation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path()
ROADMAP = ROOT / "docs/source/rust_core_roadmap.rst"
RUST_BINDING = ROOT / "bindings/rust/src/lib.rs"
PYTHON_KERNEL = ROOT / "src/innovate/kernel.py"
PYTHON_CAPABILITIES = ROOT / "src/innovate/capabilities.py"


def normalized_text(path: Path) -> str:
    """Read prose with line wrapping collapsed for stable phrase assertions."""
    return " ".join(path.read_text().split())


def test_rust_core_roadmap_documentation_is_present() -> None:
    """The Rust core trajectory should be documented as a first-class roadmap."""
    docs_root = Path("docs/source")

    assert (docs_root / "rust_core_roadmap.rst").is_file()


def test_rust_core_roadmap_names_candidate_operations_and_gates() -> None:
    """The roadmap should make migration and promotion criteria explicit."""
    roadmap = normalized_text(Path("docs/source/rust_core_roadmap.rst"))

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
    assert "same simple fitted-state payload" in roadmap
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
    roadmap = normalized_text(Path("docs/source/rust_core_roadmap.rst"))

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
        "predict_model": ("Native logistic prediction and Bass prediction", "High"),
        "simulate_model": ("Native logistic simulation and Bass simulation", "High"),
        "summarize_model": ("Native logistic summary", "Medium"),
        "diagnose_model": ("Native logistic diagnostics", "Medium"),
    }

    for operation, expected_phrases in expected_inventory.items():
        assert operation in roadmap
        for phrase in expected_phrases:
            assert phrase in roadmap


def test_rust_core_roadmap_audit_matches_current_runtime_ownership() -> None:
    """The roadmap should machine-check that Rust is not the whole core yet."""
    roadmap = normalized_text(ROADMAP)
    rust_binding = RUST_BINDING.read_text()
    python_kernel = PYTHON_KERNEL.read_text()
    capabilities = PYTHON_CAPABILITIES.read_text()

    for phrase in (
        "Audited status",
        "The core is not entirely Rust",
        "src/innovate/kernel.py",
        "bindings/rust/src/lib.rs",
        "Python reference owner",
        "packaged discovery metadata",
        "logistic ``fit_model``",
        "logistic ``summarize_model``",
        "logistic ``diagnose_model``",
        "Bass ``predict_model``/``simulate_model``",
        "Python bridge fallback path",
        "Unsupported native slices therefore remain",
        "bridge-backed",
        "A full Rust core must not be claimed",
        "every canonical operation",
        "every Python registry model family",
        "every stable payload shape",
        "covariates",
        "event splits",
        "probabilistic runtimes",
        "custom fitter options",
        "incomplete fitted states",
    ):
        assert phrase in roadmap

    for operation in (
        "discover_models",
        "fit_model",
        "predict_model",
        "simulate_model",
        "summarize_model",
        "diagnose_model",
    ):
        assert f"def {operation}" in python_kernel
        assert operation in roadmap

    for rust_anchor in (
        "pub fn discover_models_native",
        "pub fn fit_model_native",
        "pub fn predict_model_native",
        "pub fn simulate_model_native",
        "pub fn summarize_model_native",
        "pub fn diagnose_model_native",
        "fn logistic_fit_native_response",
        "fn logistic_summary_native_response",
        "fn logistic_diagnose_native_response",
        '"logistic" => logistic_native_response',
        '"bass" => bass_native_response',
        "pub fn invoke",
        "fn bridge_script_absolute_path",
        "fn kernel_pythonpath",
        "fn python_command_segments",
    ):
        assert rust_anchor in rust_binding

    assert '"unsupported_native_operation"' in rust_binding
    assert "uv run python" in rust_binding

    python_model_keys = set(re.findall(r'^\s{8}"([^"]+)": ModelCapability\(', capabilities, flags=re.MULTILINE))
    rust_native_match_keys = set(re.findall(r'"([^"]+)" => [a-z_]+_native_response', rust_binding))
    rust_native_explicit_keys = {"logistic"} if "fn logistic_fit_native_response" in rust_binding else set()
    rust_native_model_keys = rust_native_match_keys | rust_native_explicit_keys

    assert {"bass", "logistic"} <= python_model_keys
    assert {"bass", "logistic"} <= rust_native_model_keys
    assert rust_native_model_keys < python_model_keys

    non_native_model_keys = python_model_keys - rust_native_model_keys
    assert {"gompertz", "fisher_pry", "network_diffusion", "policy_hazard"} <= non_native_model_keys
    for model_key in ("gompertz", "fisher_pry", "network_diffusion", "policy_hazard"):
        assert f"``{model_key}``" in roadmap


def test_rust_core_roadmap_explicitly_rejects_full_rust_ownership() -> None:
    """The inventory should keep the partial-Rust migration state unambiguous."""
    roadmap = normalized_text(ROADMAP)

    for phrase in (
        "Python remains the primary ergonomic and reference surface",
        "the core is not fully Rust-owned today",
        "The core is not entirely Rust",
        "Python reference owner",
        "exposes Rust-native execution only for the documented slices",
        "Unsupported native slices therefore remain",
        "The core is therefore not entirely written in Rust yet",
        "Promotion remains operation by operation",
    ):
        assert phrase in roadmap


def test_rust_core_roadmap_enumerates_python_bridge_fallback_inventory() -> None:
    """Every migrated operation should document what still falls back to Python."""
    roadmap = normalized_text(ROADMAP)

    expected_fallbacks = {
        "discover_models": "Bridge discovery remains available for parity and drift checks.",
        "fit_model": "Unsupported model families, covariates, event splits, and custom fitter options fall back to the Python bridge.",
        "predict_model": "Unsupported families, covariate payloads, event splits, and incomplete fitted states fall back to the Python bridge.",
        "simulate_model": "Unsupported families, stochastic policies that are not represented in the stable payload, covariates, and event splits fall back to the bridge.",
        "summarize_model": "Unsupported families, custom diagnostics, covariates, and event splits fall back to the bridge.",
        "diagnose_model": "Unsupported families, missing diagnostic inputs, covariates, and event splits fall back to the bridge when the wrapper path is used.",
    }

    for operation, fallback in expected_fallbacks.items():
        assert operation in roadmap
        assert fallback in roadmap

    for phrase in (
        "unsupported_native_operation",
        "Public wrapper methods treat that code as recoverable",
        "dispatch the original request to the Python bridge",
        "bridge_command_failed",
    ):
        assert phrase in roadmap


def test_rust_core_roadmap_captures_cpu_memory_and_gpu_promotion_evidence() -> None:
    """Promotion gates should require CPU, memory, and GPU/XLA profiling evidence."""
    roadmap = normalized_text(ROADMAP)

    for phrase in (
        "benchmark promotion dossier",
        "Criterion output for Rust-native CPU paths",
        "Python reference timings",
        "XLA compile cost and steady-state runtime when eligible",
        "memory evidence for allocation-sensitive slices",
        "regression threshold that CI or release checks can enforce",
        "cargo-flamegraph",
        "Rust-native CPU hot paths",
        "DHAT-backed",
        "GPU profiling is not currently part of the Rust crate",
        "GPU and XLA device profiling should remain attached to the optional JAX/XLA backend",
        "keep GPU profiling with the active GPU/XLA backend until Rust owns a GPU",
    ):
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
