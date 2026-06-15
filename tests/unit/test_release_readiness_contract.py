"""Release-readiness contract and fail-closed evidence tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release_readiness import (
    CONTRACT_PATH,
    REQUIRED_EVIDENCE_IDS,
    VALID_STATUS_VALUES,
    build_readiness_report,
    evaluate_evidence,
    load_contract,
    main,
    render_text,
)


def test_release_readiness_contract_declares_required_surfaces() -> None:
    """The mature gate should cover every release-critical surface."""
    contract = load_contract()

    assert Path("docs/source/_static/release_readiness_contract.json") == CONTRACT_PATH
    assert contract["schema_version"] == 1
    assert contract["status_values"] == sorted(VALID_STATUS_VALUES)

    evidence_ids = {entry["id"] for entry in contract["required_evidence"]}
    assert evidence_ids == REQUIRED_EVIDENCE_IDS
    assert evidence_ids == {
        "python_tests",
        "coverage",
        "mutation_sampling",
        "type_checks",
        "lint_format",
        "docs_build",
        "rust_tests",
        "binding_smoke",
        "package_dry_run",
        "security_audit",
        "sbom",
        "license_inventory",
        "provenance",
        "checksums",
        "reproducibility",
        "compatibility",
    }

    for entry in contract["required_evidence"]:
        assert entry["id"]
        assert entry["owner"]
        assert entry["lane"] in {"fast", "release", "nightly", "manual"}
        assert entry["freshness_days"] > 0
        assert entry["producer"]
        assert entry["artifact"]


def test_release_readiness_evaluation_fails_closed_when_evidence_is_missing(tmp_path: Path) -> None:
    """Missing required evidence must block the release-ready state."""
    contract = load_contract()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    for entry in contract["required_evidence"][:-1]:
        artifact = evidence_dir / entry["artifact"]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"status": "pass"}) + "\n", encoding="utf-8")

    report = evaluate_evidence(contract=contract, evidence_root=evidence_dir)

    assert report["overall_status"] == "blocked"
    assert report["release_state"] == "release_candidate"
    assert report["missing_evidence"] == [contract["required_evidence"][-1]["id"]]
    assert report["status_counts"]["missing"] == 1


@pytest.mark.parametrize("status", ["missing", "stale", "fail"])
def test_release_readiness_statuses_block_release_ready(tmp_path: Path, status: str) -> None:
    """Any non-passing required evidence should block release-ready output."""
    contract = load_contract()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    blocked_id = next(iter(REQUIRED_EVIDENCE_IDS))
    for entry in contract["required_evidence"]:
        if status == "missing" and entry["id"] == blocked_id:
            continue
        artifact = evidence_dir / entry["artifact"]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"status": status if entry["id"] == blocked_id else "pass"}) + "\n",
            encoding="utf-8",
        )

    report = evaluate_evidence(contract=contract, evidence_root=evidence_dir)

    assert report["overall_status"] == "blocked"
    if status == "missing":
        assert blocked_id in report["missing_evidence"]
    elif status == "stale":
        assert blocked_id in {item["id"] for item in report["stale_evidence"]}
    else:
        assert blocked_id in {item["id"] for item in report["failing_evidence"]}


def test_build_readiness_report_uses_committed_contract() -> None:
    """The public report builder should load the committed contract by default."""
    report = build_readiness_report()

    assert report["schema_version"] == 1
    assert report["contract_path"] == str(CONTRACT_PATH)
    assert set(report["status_counts"]) >= {"pass", "missing", "stale", "fail"}


def test_release_readiness_report_passes_when_all_evidence_is_fresh(tmp_path: Path) -> None:
    """A complete fresh evidence set should produce release-ready status."""
    contract = load_contract()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    for entry in contract["required_evidence"]:
        artifact = evidence_dir / entry["artifact"]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"status": "pass", "summary": entry["id"]}) + "\n", encoding="utf-8")

    report = evaluate_evidence(contract=contract, evidence_root=evidence_dir)
    rendered = render_text(report)

    assert report["overall_status"] == "release_ready"
    assert report["release_state"] == "release_ready"
    assert report["status_counts"]["pass"] == len(REQUIRED_EVIDENCE_IDS)
    assert "overall_status: release_ready" in rendered


def test_release_readiness_report_detects_invalid_evidence_payload(tmp_path: Path) -> None:
    """Evidence artifacts must be JSON objects so status fields are auditable."""
    contract = load_contract()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    first_entry = contract["required_evidence"][0]
    artifact = evidence_dir / first_entry["artifact"]
    artifact.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(TypeError, match="Evidence artifact must be a JSON object"):
        evaluate_evidence(contract=contract, evidence_root=evidence_dir)


def test_release_readiness_contract_validation_detects_drift(tmp_path: Path) -> None:
    """The evaluator should reject committed contract drift."""
    contract = load_contract()
    drifted_ids = dict(contract)
    drifted_ids["required_evidence"] = contract["required_evidence"][1:]
    drifted_id_path = tmp_path / "drifted-ids.json"
    drifted_id_path.write_text(json.dumps(drifted_ids), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence ids drifted"):
        load_contract(drifted_id_path)

    drifted_statuses = dict(contract)
    drifted_statuses["status_values"] = ["pass"]
    drifted_status_path = tmp_path / "drifted-statuses.json"
    drifted_status_path.write_text(json.dumps(drifted_statuses), encoding="utf-8")

    with pytest.raises(ValueError, match="status values drifted"):
        load_contract(drifted_status_path)


def test_release_readiness_report_treats_non_evidence_status_as_failure(tmp_path: Path) -> None:
    """Release-state labels in evidence artifacts should not count as passing evidence."""
    contract = load_contract()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    blocked_entry = contract["required_evidence"][0]

    for entry in contract["required_evidence"]:
        artifact = evidence_dir / entry["artifact"]
        artifact.write_text(
            json.dumps({"status": "release_ready" if entry == blocked_entry else "pass"}) + "\n",
            encoding="utf-8",
        )

    report = evaluate_evidence(contract=contract, evidence_root=evidence_dir)

    assert report["overall_status"] == "blocked"
    assert {item["id"] for item in report["failing_evidence"]} == {blocked_entry["id"]}


def test_release_readiness_cli_writes_json_report(tmp_path: Path) -> None:
    """The CLI should expose a machine-readable report for local and CI use."""
    output = tmp_path / "readiness.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/release_readiness.py",
            "--json",
            "--evidence-root",
            str(tmp_path / "missing-evidence"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "blocked"
    assert "missing_evidence" in result.stdout


def test_release_readiness_cli_text_path_reports_blocked_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The in-process CLI path should print text when JSON output is not requested."""
    exit_code = main(["--evidence-root", str(tmp_path / "missing-evidence")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Release readiness" in captured.out
    assert "overall_status: blocked" in captured.out
