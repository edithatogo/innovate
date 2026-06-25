"""Release-hardening guard tests: evidence freshness, gate presence, fail-closed."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.release_readiness import (
    CONTRACT_PATH,
    DEFAULT_EVIDENCE_ROOT,
    FAILING_STATUSES,
    PASSING_STATUSES,
    REQUIRED_EVIDENCE_IDS,
    VALID_STATUS_VALUES,
    evaluate_evidence,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract_dict() -> dict[str, Any]:
    """Return the committed release-readiness contract as a dict."""
    return load_contract(CONTRACT_PATH)


@pytest.fixture
def tmp_evidence_root(tmp_path: Path) -> Path:
    """Create a temporary evidence root for testing."""
    return tmp_path / "evidence"



def _make_evidence(
    path: Path,
    status: str = "pass",
    summary: str = "All checks passed",
    python_version: str | None = "3.14",
    mtime_offset_days: float = 0.0,
) -> dict[str, Any]:
    """Write a minimal evidence artifact and return its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"status": status, "summary": summary}
    if python_version is not None:
        data["python_version"] = python_version
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    if mtime_offset_days > 0:
        past = time.time() - mtime_offset_days * 86_400
        os.utime(path, (past, past))
    return data

# ---------------------------------------------------------------------------
# Task 1: Evidence freshness tests
# ---------------------------------------------------------------------------


class TestEvidenceFreshness:
    """Evidence artifacts go stale when older than freshness_days."""

    def test_fresh_evidence_passes(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Recently written evidence should be 'pass' status."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            _make_evidence(artifact, status="pass")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "release_ready"
        assert report["status_counts"]["stale"] == 0

    def test_stale_evidence_fails(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Evidence older than freshness_days should be marked stale and block release."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            age_days = req["freshness_days"] + 1
            _make_evidence(artifact, status="pass", mtime_offset_days=age_days)

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"
        assert report["status_counts"]["stale"] == len(contract_dict["required_evidence"])
        assert report["release_state"] == "release_candidate"

    def test_freshness_days_are_reasonable(self, contract_dict: dict[str, Any]) -> None:
        """Every required evidence must have a positive freshness_days."""
        for req in contract_dict["required_evidence"]:
            assert req["freshness_days"] > 0, f"{req['id']} has non-positive freshness_days"

    def test_mixed_freshness(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Mix of fresh and stale evidence - only stale should be flagged."""
        for i, req in enumerate(contract_dict["required_evidence"]):
            artifact = tmp_evidence_root / req["artifact"]
            if i % 2 == 0:
                _make_evidence(artifact, status="pass")
            else:
                _make_evidence(artifact, status="pass", mtime_offset_days=req["freshness_days"] + 1)

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"
        assert report["status_counts"]["stale"] > 0
        assert report["status_counts"]["pass"] > 0

# ---------------------------------------------------------------------------
# Task 2: Required gate presence tests
# ---------------------------------------------------------------------------


class TestRequiredGatePresence:
    """All required gates must be present in the contract and have valid config."""

    def test_all_required_evidence_ids_present(self, contract_dict: dict[str, Any]) -> None:
        """The committed contract must contain all required evidence IDs."""
        evidence_ids = {entry["id"] for entry in contract_dict["required_evidence"]}
        missing = REQUIRED_EVIDENCE_IDS - evidence_ids
        extra = evidence_ids - REQUIRED_EVIDENCE_IDS
        assert evidence_ids == REQUIRED_EVIDENCE_IDS, (
            f"Missing: {missing}, Extra: {extra}"
        )

    def test_no_duplicate_evidence_ids(self, contract_dict: dict[str, Any]) -> None:
        """Evidence IDs must be unique in the contract."""
        evidence_ids = [entry["id"] for entry in contract_dict["required_evidence"]]
        assert len(evidence_ids) == len(set(evidence_ids)), "Duplicate evidence IDs found"

    def test_each_evidence_has_required_fields(self, contract_dict: dict[str, Any]) -> None:
        """Each evidence entry must have id, owner, lane, freshness_days, producer, artifact."""
        required_fields = {"id", "owner", "lane", "freshness_days", "producer", "artifact"}
        for entry in contract_dict["required_evidence"]:
            missing = required_fields - set(entry.keys())
            assert not missing, f"{entry['id']} missing fields: {missing}"

    def test_valid_lane_values(self, contract_dict: dict[str, Any]) -> None:
        """Lane must be one of: fast, nightly, monthly."""
        valid_lanes = {"fast", "nightly", "release"}
        for entry in contract_dict["required_evidence"]:
            assert entry["lane"] in valid_lanes, f"{entry['id']} has invalid lane: {entry['lane']}"

    def test_valid_status_values_defined(self, contract_dict: dict[str, Any]) -> None:
        """Contract status_values must match the evaluator's valid set."""
        assert set(contract_dict["status_values"]) == VALID_STATUS_VALUES

    def test_contract_has_schema_version(self, contract_dict: dict[str, Any]) -> None:
        """Contract should include a schema_version field."""
        assert contract_dict.get("schema_version") == 1

# ---------------------------------------------------------------------------
# Task 3: Release-ready fail-closed behavior tests
# ---------------------------------------------------------------------------


class TestReleaseReadyFailClosed:
    """Release readiness must fail closed - blocked when anything is wrong."""

    def test_all_passing_is_release_ready(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """All present, fresh, and passing evidence = release_ready."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            _make_evidence(artifact, status="pass")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "release_ready"
        assert report["release_state"] == "release_ready"

    def test_missing_evidence_blocks(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Missing evidence artifacts must block release_ready."""
        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"
        assert report["status_counts"]["missing"] > 0
        assert report["release_state"] == "release_candidate"

    def test_failing_evidence_blocks(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Evidence with fail/blocked status must block release."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            _make_evidence(artifact, status="fail")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"
        assert report["status_counts"]["fail"] > 0

    def test_single_missing_blocks_overall(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Even one missing evidence artifact should make overall_status blocked."""
        reqs = contract_dict["required_evidence"]
        for i, req in enumerate(reqs):
            if i > 0:
                artifact = tmp_evidence_root / req["artifact"]
                _make_evidence(artifact, status="pass")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"
        assert report["status_counts"]["missing"] >= 1

    def test_fail_closed_on_unknown_status(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Evidence with unknown status should be treated as failing."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            _make_evidence(artifact, status="unknown")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"

    def test_fail_closed_on_missing_python_version(
        self, contract_dict: dict[str, Any], tmp_evidence_root: Path
    ) -> None:
        """Evidence missing required Python version must fail."""
        for req in contract_dict["required_evidence"]:
            artifact = tmp_evidence_root / req["artifact"]
            python_version = req.get("required_python_version")
            if python_version:
                _make_evidence(artifact, status="pass", python_version="3.13")
            else:
                _make_evidence(artifact, status="pass")

        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        assert report["overall_status"] == "blocked"

    def test_report_includes_status_counts(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Report should always include status_counts with all four categories."""
        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        for key in ("pass", "missing", "stale", "fail"):
            assert key in report["status_counts"]

    def test_report_lists_missing_evidence_ids(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """Missing evidence should list their IDs in the report."""
        report = evaluate_evidence(contract=contract_dict, evidence_root=tmp_evidence_root)
        expected_ids = {req["id"] for req in contract_dict["required_evidence"]}
        assert set(report["missing_evidence"]) == expected_ids

    def test_command_line_exit_code_fail_closed(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """CLI should return non-zero when blocked without --allow-blocked."""
        from scripts.release_readiness import main

        exit_code = main(["--json", "--evidence-root", str(tmp_evidence_root)])
        assert exit_code == 1  # blocked with missing evidence

    def test_command_line_allow_blocked(self, contract_dict: dict[str, Any], tmp_evidence_root: Path) -> None:
        """CLI should return zero with --allow-blocked even when blocked."""
        from scripts.release_readiness import main

        exit_code = main(["--json", "--evidence-root", str(tmp_evidence_root), "--allow-blocked"])
        assert exit_code == 0
