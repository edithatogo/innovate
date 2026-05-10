"""Tests for the machine-readable Rust core migration inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_OPERATIONS = {
    "discover_models",
    "fit_model",
    "predict_model",
    "simulate_model",
    "summarize_model",
    "diagnose_model",
}
INVENTORY_PATH = Path("docs/source/_static/rust_core_migration_inventory.json")
DOSSIER_PATH = Path("docs/source/_static/rust_core_promotion_dossier_example.json")
BASS_DOSSIER_PATH = Path("docs/source/_static/rust_core_promotion_dossier_bass_example.json")
ROADMAP_PATH = Path("docs/source/rust_core_roadmap.rst")
BENCH_PATH = Path("bindings/rust/benches/native_kernel.rs")
OPERATIONS_TEST_PATH = Path("bindings/rust/tests/operations.rs")
NATIVE_DISCOVERY_TEST_PATH = Path("bindings/rust/tests/native_discovery.rs")
BENCH_RESULTS_PATH = Path("docs/source/_static/rust_core_native_benchmark_results.json")
REQUIRED_ENTRY_FIELDS = {
    "operation",
    "model_slice",
    "current_owner",
    "fallback_status",
    "native_scope",
    "fallback_scope",
    "python_reference_scope",
    "profiling_requirements",
    "promotion_blockers",
}
REQUIRED_PROMOTION_GATES = {
    "parity",
    "schema_compatibility",
    "error_mapping",
    "benchmark_evidence",
    "memory_evidence",
    "binding_smoke",
}
REQUIRED_BINDINGS = {"python", "rust", "r", "julia", "typescript", "go", "csharp"}


def load_inventory() -> dict[str, Any]:
    """Load the Rust migration inventory artifact."""
    return json.loads(INVENTORY_PATH.read_text())


def load_dossier() -> dict[str, Any]:
    """Load the Rust promotion dossier example artifact."""
    return json.loads(DOSSIER_PATH.read_text())


def load_bass_dossier() -> dict[str, Any]:
    """Load the Bass-specific Rust promotion dossier example artifact."""
    return json.loads(BASS_DOSSIER_PATH.read_text())


def normalized_text(path: Path) -> str:
    """Read prose with line wrapping collapsed for stable phrase assertions."""
    return " ".join(path.read_text().split())


def test_rust_migration_inventory_declares_required_fields_and_enums() -> None:
    """Each inventory row should use the documented field names and enum values."""
    inventory = load_inventory()

    assert inventory["schema_version"] == 1
    assert inventory["gap_track"] == "rust_core_migration_completion_20260511"
    assert set(inventory["owner_values"]) == {"rust_native", "python_bridge", "python_reference"}
    assert set(inventory["fallback_status_values"]) == {
        "native_default_no_fallback_needed",
        "native_default_python_bridge_fallback",
        "python_bridge_default",
        "python_reference_only",
    }
    assert set(inventory["promotion_state_values"]) == {
        "native_default_guarded",
        "native_candidate_needs_evidence",
        "bridge_default_pending_migration",
        "bridge_default_explicit_exception",
        "python_reference_boundary",
    }
    assert inventory["inventory"]

    owner_values = set(inventory["owner_values"])
    fallback_values = set(inventory["fallback_status_values"])
    for entry in inventory["inventory"]:
        assert set(entry) >= REQUIRED_ENTRY_FIELDS
        assert entry["current_owner"] in owner_values
        assert entry["fallback_status"] in fallback_values
        assert isinstance(entry["profiling_requirements"], list)
        assert isinstance(entry["promotion_blockers"], list)
        assert all(isinstance(requirement, str) and requirement for requirement in entry["profiling_requirements"])
        assert all(isinstance(blocker, str) and blocker for blocker in entry["promotion_blockers"])


def test_rust_migration_inventory_covers_canonical_operations_and_mixed_ownership() -> None:
    """The inventory should cover all canonical operations without claiming full Rust ownership."""
    inventory = load_inventory()
    entries = inventory["inventory"]
    operations = {entry["operation"] for entry in entries}
    owners = {entry["current_owner"] for entry in entries}
    fallback_statuses = {entry["fallback_status"] for entry in entries}

    assert operations >= CANONICAL_OPERATIONS
    assert "rust_native" in owners
    assert "python_bridge" in owners
    assert "python_reference" in owners
    assert "native_default_python_bridge_fallback" in fallback_statuses
    assert "python_bridge_default" in fallback_statuses
    assert any(entry["current_owner"] == "rust_native" for entry in entries)
    assert any(entry["current_owner"] == "python_bridge" for entry in entries)
    assert any(entry["operation"] == "all_kernel_operations" for entry in entries)


def test_rust_migration_inventory_records_native_and_fallback_slices() -> None:
    """The inventory should include concrete Rust-native slices and Python bridge slices."""
    entries = load_inventory()["inventory"]
    native_slices = {
        (entry["operation"], entry["model_slice"]) for entry in entries if entry["current_owner"] == "rust_native"
    }
    bridge_slices = {
        (entry["operation"], entry["model_slice"]) for entry in entries if entry["current_owner"] == "python_bridge"
    }

    assert ("discover_models", "all_packaged_discovery_metadata") in native_slices
    assert ("fit_model", "gompertz_simple_positive_observations") in native_slices
    assert ("predict_model", "gompertz_simple_fitted_state") in native_slices
    assert ("simulate_model", "gompertz_simple_fitted_state") in native_slices
    assert ("summarize_model", "gompertz_simple_fitted_state") in native_slices
    assert ("diagnose_model", "gompertz_simple_fitted_state") in native_slices
    assert ("predict_model", "bass_simple_fitted_state") in native_slices
    assert ("simulate_model", "bass_simple_fitted_state") in native_slices
    assert ("fit_model", "bass_and_other_model_families") in bridge_slices
    assert ("predict_model", "other_model_families_or_unsupported_payloads") in bridge_slices
    assert ("diagnose_model", "bass_and_other_model_families") in bridge_slices
    assert {
        (entry["operation"], entry["model_slice"])
        for entry in entries
        if entry["promotion_state"] == "bridge_default_explicit_exception"
    } == {
        ("fit_model", "bass_and_other_model_families"),
        ("predict_model", "other_model_families_or_unsupported_payloads"),
        ("simulate_model", "other_model_families_or_unsupported_payloads"),
        ("summarize_model", "bass_and_other_model_families"),
        ("diagnose_model", "bass_and_other_model_families"),
    }

    for operation, _model_slice in bridge_slices:
        assert operation in CANONICAL_OPERATIONS


def test_rust_migration_inventory_is_execution_grade_backlog() -> None:
    """Every migration slice should carry operation-level execution metadata."""
    inventory = load_inventory()
    entries = inventory["inventory"]

    assert inventory["migration_phase_values"]
    assert inventory["promotion_state_values"]

    phase_values = set(inventory["migration_phase_values"])
    state_values = set(inventory["promotion_state_values"])

    for entry in entries:
        assert entry["migration_phase"] in phase_values
        assert entry["promotion_state"] in state_values
        assert isinstance(entry["depends_on"], list)
        assert all(isinstance(dependency, str) and dependency for dependency in entry["depends_on"])
        assert isinstance(entry["unsupported_payload_shapes"], list)
        assert entry["python_reference_scope"]
        assert isinstance(entry["promotion_path"], list)
        assert len(entry["promotion_path"]) >= 3

    phase_by_key = {(entry["operation"], entry["model_slice"]): entry["migration_phase"] for entry in entries}
    assert phase_by_key[("discover_models", "all_packaged_discovery_metadata")] == "phase_0_native_guardrails"
    assert phase_by_key[("fit_model", "gompertz_simple_positive_observations")] == "phase_2_logistic_expansion"
    assert phase_by_key[("predict_model", "gompertz_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("simulate_model", "gompertz_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("summarize_model", "gompertz_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("diagnose_model", "gompertz_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("predict_model", "bass_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("simulate_model", "bass_simple_fitted_state")] == "phase_1_default_hardening"
    assert phase_by_key[("fit_model", "bass_and_other_model_families")] == "phase_3_model_family_migration"
    assert (
        phase_by_key[
            ("all_kernel_operations", "probabilistic_runtimes_uncertainty_and_python_object_internals")
        ]
        == "phase_4_reference_boundary_review"
    )


def test_rust_migration_inventory_defines_operation_promotion_gates() -> None:
    """Promotion gates should be concrete enough to execute by operation family."""
    entries = load_inventory()["inventory"]

    for entry in entries:
        gates = entry["promotion_gates"]
        assert REQUIRED_PROMOTION_GATES <= set(gates)
        for gate_name in REQUIRED_PROMOTION_GATES:
            gate = gates[gate_name]
            assert gate["status"] in {"passed", "required", "not_applicable"}
            assert gate["evidence"]
            assert gate["required_before_default"] in {True, False}

        if entry["current_owner"] == "rust_native":
            assert gates["parity"]["status"] == "passed"
            assert gates["schema_compatibility"]["status"] == "passed"
            assert gates["error_mapping"]["status"] == "passed"
            assert gates["benchmark_evidence"]["status"] == "passed"
            assert gates["benchmark_evidence"]["required_before_default"] is True

        if entry["current_owner"] != "rust_native":
            assert gates["parity"]["required_before_default"] is True
            assert gates["schema_compatibility"]["required_before_default"] is True
            assert gates["error_mapping"]["required_before_default"] is True


def test_rust_migration_inventory_defines_binding_smoke_contracts() -> None:
    """Every promoted operation should specify thin-binding smoke expectations."""
    entries = load_inventory()["inventory"]

    for entry in entries:
        smoke = entry["binding_smoke"]
        assert set(smoke["required_bindings"]) == REQUIRED_BINDINGS
        assert smoke["operation"] == entry["operation"]
        assert smoke["contract"] in {
            "schema_envelope",
            "request_response_roundtrip",
            "fallback_error_contract",
        }
        assert smoke["required_before_default"] in {True, False}
        if entry["current_owner"] == "rust_native":
            assert smoke["required_before_default"] is True


def test_rust_migration_inventory_records_benchmark_and_memory_evidence_commands() -> None:
    """Benchmark and profiling requirements should point at concrete repo commands."""
    entries = load_inventory()["inventory"]
    combined = json.dumps(entries)

    for command in (
        "cargo bench --bench native_kernel",
        "bindings/rust/scripts/profile_native_kernels.sh",
        "bindings/rust/scripts/profile_memory_native_kernels.sh",
        "JAX_PLATFORM_NAME=cpu",
        "JAX_PLATFORM_NAME=gpu",
    ):
        assert command in combined

    for entry in entries:
        evidence = entry["evidence_commands"]
        assert "parity" in evidence
        assert "schema" in evidence
        assert "benchmark" in evidence
        assert "memory" in evidence
        assert "binding_smoke" in evidence


def test_rust_migration_inventory_profiles_required_promotion_evidence_categories() -> None:
    """Inventory and roadmap should preserve the promotion dossier evidence categories."""
    entries = load_inventory()["inventory"]
    requirements = " ".join(requirement for entry in entries for requirement in entry["profiling_requirements"])
    blockers = " ".join(blocker for entry in entries for blocker in entry["promotion_blockers"])
    roadmap = normalized_text(ROADMAP_PATH)

    for phrase in (
        "Criterion benchmark coverage",
        "CPU flamegraph",
        "Memory profiling",
        "Rust CPU timing",
        "bridge fallback rate",
        "XLA eligibility",
    ):
        assert phrase in requirements + " " + blockers

    for phrase in (
        "promotion dossier",
        "Criterion output for Rust-native CPU paths",
        "Python reference timings",
        "XLA compile cost and steady-state runtime when eligible",
        "memory evidence for allocation-sensitive slices",
        "regression threshold that CI or release checks can enforce",
    ):
        assert phrase in roadmap


def test_rust_promotion_dossier_example_records_required_evidence_sections() -> None:
    """The promotion dossier example should preserve the required evidence structure."""
    dossier = load_dossier()

    assert dossier["schema_version"] == 1
    assert dossier["dossier_type"] == "rust_core_promotion_dossier"
    assert dossier["evidence_state"] == "template_example_not_release_evidence"
    assert dossier["slice"]["operation"] in CANONICAL_OPERATIONS
    assert dossier["slice"]["current_owner"] in {"rust_native", "python_bridge", "python_reference"}
    assert dossier["slice"]["target_owner"] in {"rust_native", "python_bridge", "python_reference"}
    assert dossier["slice"]["fallback_status"] in {
        "native_default_no_fallback_needed",
        "native_default_python_bridge_fallback",
        "python_bridge_default",
        "python_reference_only",
    }

    required_sections = {
        "promotion_decision",
        "parity",
        "schema_compatibility",
        "error_mapping",
        "cpu_benchmark",
        "memory_profiling",
        "xla_gpu_eligibility",
        "fallback_rate",
        "binding_smoke",
        "release_record",
    }
    assert required_sections <= set(dossier)

    assert "regression_threshold" in dossier["cpu_benchmark"]["metrics"]
    assert "allocation_regression_threshold" in dossier["memory_profiling"]["metrics"]
    assert "XLA compile cost" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "XLA steady-state runtime" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "Rust-native CPU runtime" in dossier["xla_gpu_eligibility"]["required_comparison_when_eligible"]
    assert "python_bridge_fallback_requests" in dossier["fallback_rate"]["metrics"]
    assert {"rust", "python"} <= set(dossier["binding_smoke"]["required_bindings"])


def test_rust_bass_promotion_dossier_example_records_bass_evidence() -> None:
    """The Bass promotion dossier example should point at the Bass native slice."""
    dossier = load_bass_dossier()

    assert dossier["schema_version"] == 1
    assert dossier["dossier_type"] == "rust_core_promotion_dossier"
    assert dossier["evidence_state"] == "template_example_not_release_evidence"
    assert dossier["slice"]["operation"] == "predict_model"
    assert dossier["slice"]["model_slice"] == "bass_simple_fitted_state"
    assert dossier["slice"]["current_owner"] == "rust_native"
    assert "native_logistic_kernel/predict_model_native/bass" in dossier["cpu_benchmark"]["required_artifacts"][0]
    assert "native_bass_prediction_matches_python_bridge_contract" in dossier["parity"]["test_evidence"][0]["command"]
    assert "native_bass_reports_structured_errors_for_invalid_or_unsupported_shapes" in dossier["error_mapping"]["test_evidence"][1]["command"]


def test_rust_native_benchmark_harness_includes_bass_cases() -> None:
    """The Rust Criterion harness should benchmark the native promoted slices."""
    bench = BENCH_PATH.read_text()
    results = json.loads(BENCH_RESULTS_PATH.read_text())

    assert "fn bass_predict_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn bass_simulate_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn gompertz_fit_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn gompertz_predict_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn gompertz_simulate_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn gompertz_summary_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn gompertz_diagnose_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn fisher_pry_fit_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn fisher_pry_predict_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn fisher_pry_simulate_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn fisher_pry_summary_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert "fn fisher_pry_diagnose_request(binding: &KernelBinding) -> KernelRequest" in bench
    assert 'BenchmarkId::new("fit_model_native", "gompertz")' in bench
    assert 'BenchmarkId::new("fit_model_native", "fisher_pry")' in bench
    assert 'BenchmarkId::new("predict_model_native", "gompertz")' in bench
    assert 'BenchmarkId::new("predict_model_native", "fisher_pry")' in bench
    assert 'BenchmarkId::new("simulate_model_native", "gompertz")' in bench
    assert 'BenchmarkId::new("simulate_model_native", "fisher_pry")' in bench
    assert 'BenchmarkId::new("summarize_model_native", "gompertz")' in bench
    assert 'BenchmarkId::new("summarize_model_native", "fisher_pry")' in bench
    assert 'BenchmarkId::new("diagnose_model_native", "gompertz")' in bench
    assert 'BenchmarkId::new("diagnose_model_native", "fisher_pry")' in bench
    assert 'BenchmarkId::new("predict_model_native", "bass")' in bench
    assert 'BenchmarkId::new("simulate_model_native", "bass")' in bench
    assert results["schema_version"] == 1
    assert results["source_command"] == "cargo bench --manifest-path bindings/rust/Cargo.toml --bench native_kernel"
    assert len(results["benchmarks"]) == 17
    assert {entry["model_key"] for entry in results["benchmarks"]} >= {"logistic", "gompertz", "fisher_pry", "bass"}


def test_rust_native_operations_suite_covers_all_promoted_slices() -> None:
    """The Rust operations tests should cover every promoted native slice and fallback rule."""
    operations = normalized_text(OPERATIONS_TEST_PATH)
    native_discovery = normalized_text(NATIVE_DISCOVERY_TEST_PATH)

    expected_native_tests = {
        "native_logistic_fit_matches_python_bridge_contract",
        "native_logistic_prediction_matches_python_bridge_contract",
        "native_logistic_simulation_matches_python_bridge_contract",
        "native_logistic_summary_matches_python_bridge_contract",
        "native_logistic_diagnose_matches_python_bridge_contract",
        "native_gompertz_fit_matches_python_bridge_contract",
        "native_gompertz_prediction_matches_python_bridge_contract",
        "native_gompertz_simulation_matches_python_bridge_contract",
        "native_gompertz_summary_matches_python_bridge_contract",
        "native_gompertz_diagnose_matches_python_bridge_contract",
        "native_fisher_pry_fit_matches_python_bridge_contract",
        "native_fisher_pry_prediction_matches_python_bridge_contract",
        "native_fisher_pry_simulation_matches_python_bridge_contract",
        "native_fisher_pry_summary_matches_python_bridge_contract",
        "native_fisher_pry_diagnose_matches_python_bridge_contract",
        "native_bass_prediction_matches_python_bridge_contract",
        "native_bass_simulation_matches_python_bridge_contract",
        "native_bass_reports_structured_errors_for_invalid_or_unsupported_shapes",
        "native_prediction_falls_back_to_bridge_for_non_native_models",
        "native_simulation_falls_back_to_bridge_for_non_native_models",
        "native_summary_and_diagnose_fall_back_to_bridge_for_non_native_models",
        "native_fallback_paths_emit_tracing_events",
    }

    for test_name in expected_native_tests:
        assert test_name in operations

    for discovery_test in {
        "native_discovery_manifest_is_packaged_and_decodable",
        "native_discovery_matches_python_bridge_metadata",
        "native_discovery_reports_structured_decode_errors",
        "native_discovery_reports_missing_results_as_bridge_failures",
    }:
        assert discovery_test in native_discovery
