"""Regression tests for external submission blocker closure."""

from __future__ import annotations

import json
from pathlib import Path


TARGET_INVENTORY = Path("docs/source/_static/external_submission_target_inventory.json")
REGISTRY_RECEIPTS = Path("docs/source/_static/registry_submission_receipts.json")
HPC_BLOCKERS = Path(
    "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json"
)

PACKAGE_TARGETS = {
    "python_pypi",
    "typescript_npm",
    "rust_crates_io",
    "r_r_universe",
    "r_cran",
    "julia_general",
    "go_modules",
    "csharp_nuget",
}

HPC_TARGETS = {"spack", "easybuild", "hpsf", "e4s"}

COMMUNITY_TARGETS = {
    "pyopensci",
    "ropensci",
    "joss",
    "numfocus",
    "pypa",
    "apache_arrow",
    "dotnet_foundation",
    "julia_community",
    "r_community",
    "scikit_learn_contrib",
}


def _target_inventory() -> dict:
    return json.loads(TARGET_INVENTORY.read_text(encoding="utf-8"))


def test_external_submission_inventory_covers_every_required_target() -> None:
    """Every package, HPC, and community target should be normalized."""
    inventory = _target_inventory()
    targets = {entry["target_id"]: entry for entry in inventory["targets"]}

    assert inventory["schema_version"] == 1
    assert set(inventory["target_groups"]["package_manager"]) == PACKAGE_TARGETS
    assert set(inventory["target_groups"]["hpc"]) == HPC_TARGETS
    assert set(inventory["target_groups"]["scientific_community"]) == COMMUNITY_TARGETS
    assert set(targets) == PACKAGE_TARGETS | HPC_TARGETS | COMMUNITY_TARGETS


def test_external_submission_targets_have_closure_fields() -> None:
    """Targets need current status, owner, evidence, and next action."""
    inventory = _target_inventory()
    allowed_statuses = set(inventory["status_values"])

    for entry in inventory["targets"]:
        assert entry["status"] in allowed_statuses
        assert entry["owner"], entry["target_id"]
        assert entry["evidence"], entry["target_id"]
        assert entry["next_action"], entry["target_id"]
        assert entry["group"] in inventory["target_groups"]
        if entry["status"] == "blocked":
            assert "blocked" not in entry["next_action"].lower()
        if entry["status"] == "submitted":
            assert any(str(link).startswith("https://") for link in entry["evidence"])


def test_registry_receipts_match_current_package_and_hpc_closure_states() -> None:
    """Receipt bundles should not preserve generic blocked states after closure."""
    inventory = {
        entry["target_id"]: entry for entry in _target_inventory()["targets"]
    }
    receipts = json.loads(REGISTRY_RECEIPTS.read_text(encoding="utf-8"))

    submitted = {entry["target_id"]: entry for entry in receipts["submitted_targets"]}
    pending = {entry["target_id"]: entry for entry in receipts["pending_targets"]}

    for target_id in PACKAGE_TARGETS:
        state = inventory[target_id]["status"]
        if state == "submitted":
            assert target_id in submitted
            assert submitted[target_id]["receipt_url"].startswith("https://")
        else:
            assert target_id in pending
            assert pending[target_id]["status"] == state
            assert pending[target_id].get("owner")
            assert pending[target_id].get("next_action")

    for target_id in HPC_TARGETS:
        assert target_id in pending
        assert pending[target_id]["status"] == inventory[target_id]["status"]
        assert pending[target_id].get("owner")
        assert pending[target_id].get("evidence")
        assert pending[target_id].get("next_action")


def test_hpc_blockers_are_current_and_do_not_hide_ready_targets() -> None:
    """Only actually blocked HPC targets should appear in the blocker bundle."""
    inventory = {
        entry["target_id"]: entry for entry in _target_inventory()["targets"]
    }
    blocker_bundle = json.loads(HPC_BLOCKERS.read_text(encoding="utf-8"))
    blockers = {entry["target_id"]: entry for entry in blocker_bundle["blockers"]}

    expected_blocked = {
        target_id
        for target_id in HPC_TARGETS
        if inventory[target_id]["status"] == "blocked"
    }
    assert set(blockers) == expected_blocked

    for target_id, blocker in blockers.items():
        assert blocker["status"] == "blocked"
        assert blocker.get("owner")
        assert blocker.get("evidence")
        assert blocker.get("next_action")


def test_docs_do_not_overclaim_submission_or_acceptance_without_receipts() -> None:
    """Docs should distinguish submitted, accepted, ready, blocked, and deferred."""
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path("docs/source/registry_submission_receipts.rst"),
            Path("docs/source/hpc_submission_packet.rst"),
            Path("docs/source/community_submission_readiness.rst"),
            Path("docs/source/submission_readiness_dossiers.rst"),
            Path("docs/astro-site/src/content/docs/operations/registry-submissions.md"),
            Path("docs/astro-site/src/content/docs/operations/hpc-submission-packet.md"),
            Path("docs/astro-site/src/content/docs/operations/community-readiness.md"),
        )
    ).lower()

    forbidden_claims = (
        "spack: submitted",
        "easybuild: submitted",
        "hpsf: submitted",
        "e4s: submitted",
        "cran: submitted",
        "julia general accepted",
        "julia general merged",
    )
    for claim in forbidden_claims:
        assert claim not in docs

    assert "external_submission_target_inventory.json" in docs
