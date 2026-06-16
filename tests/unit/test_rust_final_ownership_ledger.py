"""Regression tests for the final Rust ownership ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from innovate.kernel import discover_models

LEDGER_PATH = Path("docs/source/_static/rust_final_ownership_ledger.json")
MODEL_FAMILY_COVERAGE = Path("docs/source/_static/rust_native_model_family_coverage.json")
PAYLOAD_SHAPE_COVERAGE = Path("docs/source/_static/rust_native_payload_shape_coverage.json")
OPERATION_GAP_INVENTORY = Path("docs/source/_static/rust_native_operation_gap_inventory.json")
FULL_OWNERSHIP_VALIDATION = Path("docs/source/_static/rust_full_ownership_validation.json")
NATIVE_BENCHMARK_RESULTS = Path("docs/source/_static/rust_core_native_benchmark_results.json")
ROADMAP_DOCS = [
    Path("docs/source/rust_core_roadmap.rst"),
    Path("docs/astro-site/src/content/docs/operations/rust-core.md"),
    Path("docs/source/_static/vision_roadmap_status_inventory.json"),
]

ALLOWED_LEDGER_STATUSES = {
    "rust_native_promoted",
    "retain_outside_core",
    "requires_design_decision",
}
ALLOWED_RELEASE_CLAIM_STATES = {
    "claimable_native",
    "claimable_external_boundary",
    "not_claimable_until_promoted",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_final_ownership_ledger_covers_all_model_families_payloads_and_operations() -> None:
    """The final ledger must own every current Rust-ownership surface exactly once."""
    ledger = _load_json(LEDGER_PATH)
    model_coverage = _load_json(MODEL_FAMILY_COVERAGE)
    payload_coverage = _load_json(PAYLOAD_SHAPE_COVERAGE)
    operation_inventory = _load_json(OPERATION_GAP_INVENTORY)

    registry_model_keys = {record.key for record in discover_models().models}
    ledger_models = {entry["id"] for entry in ledger["model_families"]}
    coverage_models = {entry["model_key"] for entry in model_coverage["families"]}
    assert ledger_models == registry_model_keys == coverage_models

    ledger_payloads = {entry["id"] for entry in ledger["payload_shapes"]}
    coverage_payloads = {entry["payload_shape"] for entry in payload_coverage["payload_shapes"]}
    assert ledger_payloads == coverage_payloads

    ledger_operations = {entry["id"] for entry in ledger["canonical_operations"]}
    operation_names = set(operation_inventory["canonical_operations"])
    assert ledger_operations == operation_names


def test_final_ownership_ledger_entries_have_owner_rationale_evidence_and_revisit_policy() -> None:
    """Every ownership entry needs enough information to support release claims."""
    ledger = _load_json(LEDGER_PATH)

    for section in ("model_families", "payload_shapes", "canonical_operations"):
        for entry in ledger[section]:
            assert entry["status"] in ALLOWED_LEDGER_STATUSES
            assert entry["owner"]
            assert entry["rationale"]
            assert entry["evidence"]
            assert entry["release_claim_state"] in ALLOWED_RELEASE_CLAIM_STATES
            assert entry["revisit_condition"]

            for evidence_path in entry["evidence"]:
                assert Path(evidence_path).exists(), evidence_path

            if entry["status"] != "rust_native_promoted":
                assert entry["release_claim_state"] != "claimable_native"


def test_final_ownership_ledger_preserves_current_claim_boundaries() -> None:
    """The ledger must not silently convert bridge/reference boundaries into native claims."""
    ledger = _load_json(LEDGER_PATH)

    models = {entry["id"]: entry for entry in ledger["model_families"]}
    payloads = {entry["id"]: entry for entry in ledger["payload_shapes"]}

    for model_key in ("composite", "multi_product"):
        assert models[model_key]["status"] == "requires_design_decision"
        assert models[model_key]["release_claim_state"] == "not_claimable_until_promoted"

    for model_key in (
        "complementary_goods",
        "hierarchical",
        "latent_process",
        "lotka_volterra",
        "mixture",
        "network_diffusion",
        "policy_hazard",
        "regime_switching",
    ):
        assert models[model_key]["status"] == "retain_outside_core"
        assert models[model_key]["release_claim_state"] == "claimable_external_boundary"

    for payload_shape in ("covariates", "event_splits", "incomplete_fitted_state"):
        assert payloads[payload_shape]["status"] == "requires_design_decision"

    for payload_shape in (
        "graph_or_agent_state",
        "stochastic_simulation_policy",
        "uncertainty_or_posterior",
    ):
        assert payloads[payload_shape]["status"] == "retain_outside_core"


def test_stable_payload_shapes_are_fail_closed_against_unowned_entries() -> None:
    """Stable payloads must be Rust-owned or fail closed before release claims."""
    ledger = _load_json(LEDGER_PATH)

    guardrails = ledger["fail_closed_contracts"]["stable_payload_shapes"]
    assert guardrails["default"] == "fail_release_claim"
    assert guardrails["missing_entry"] == "fail_release_claim"
    assert guardrails["missing_schema_fixture"] == "fail_release_claim"
    assert guardrails["allowed_native_statuses"] == ["rust_native_promoted"]

    payload_entries = {entry["id"]: entry for entry in ledger["payload_shapes"]}
    stable_payloads = [
        entry for entry in _load_json(PAYLOAD_SHAPE_COVERAGE)["payload_shapes"] if entry["status"] == "stable"
    ]
    assert stable_payloads

    for payload in stable_payloads:
        ledger_entry = payload_entries[payload["payload_shape"]]
        assert ledger_entry["status"] == "rust_native_promoted"
        assert ledger_entry["release_claim_state"] == "claimable_native"


def test_promoted_model_families_are_fail_closed_against_missing_operations() -> None:
    """Promoted model families must list the Rust operations they support."""
    ledger = _load_json(LEDGER_PATH)

    guardrails = ledger["fail_closed_contracts"]["promoted_model_operations"]
    required_operations = set(guardrails["required_operations"])
    assert required_operations == {
        "fit_model",
        "predict_model",
        "simulate_model",
        "summarize_model",
        "diagnose_model",
    }
    assert guardrails["missing_operation"] == "fail_release_claim"
    assert guardrails["missing_binding_smoke"] == "fail_release_claim"

    for entry in ledger["model_families"]:
        if entry["status"] != "rust_native_promoted":
            continue
        operations = set(entry["native_operations"])
        assert operations == required_operations


def test_docs_full_ownership_claims_are_fail_closed_by_ledger() -> None:
    """Docs must not claim full Rust ownership until the ledger allows it."""
    ledger = _load_json(LEDGER_PATH)
    guardrails = ledger["fail_closed_contracts"]["documentation_claims"]

    assert guardrails["full_rust_ownership_claim_allowed"] is False
    assert guardrails["missing_ledger_reference"] == "fail_docs_claim"
    assert guardrails["overclaim_state"] == "fail_docs_claim"

    overclaim_phrases = {
        "fully rust-owned",
        "full rust ownership is complete",
        "all model families are rust-native",
        "all stable payload shapes are rust-native",
    }
    for path in ROADMAP_DOCS:
        text = path.read_text().lower()
        assert "rust_final_ownership_ledger.json" in text
        assert not any(phrase in text for phrase in overclaim_phrases)


def test_full_rust_ownership_validation_is_fail_closed_with_benchmark_evidence() -> None:
    """Release evidence should allow only the narrower promoted-slice claim."""
    validation = _load_json(FULL_OWNERSHIP_VALIDATION)
    benchmarks = _load_json(NATIVE_BENCHMARK_RESULTS)

    assert validation["decision"] == "full_rust_ownership_not_claimed"
    assert validation["release_claim_gate"]["full_rust_ownership_claim_allowed"] is False
    assert validation["release_claim_gate"]["allowed_claim"] == (
        "promoted deterministic Rust-native slices have benchmark and binding dispatch evidence"
    )
    assert validation["benchmark_evidence"]["source"] == str(NATIVE_BENCHMARK_RESULTS)
    assert validation["benchmark_evidence"]["promoted_slice_count"] == 25
    assert validation["benchmark_evidence"]["regression_policy"] == (
        "fail_release_claim_when_any_promoted_native_slice_exceeds_threshold_without_a_waiver"
    )
    assert benchmarks["release_claim_policy"]["full_rust_ownership_claim_allowed"] is False
    assert benchmarks["regression_thresholds"]["max_upper_bound_regression_ratio"] == 1.25
