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
