"""Tests for offline supply-chain release evidence generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.release_readiness import build_readiness_report
from scripts.release_supply_chain import EVIDENCE_IDS, build_supply_chain_evidence

EVIDENCE_ROOT = Path("docs/source/_static/release_readiness/evidence")


def test_supply_chain_evidence_generator_writes_required_artifacts(tmp_path: Path) -> None:
    """Supply-chain evidence should be generated without secrets or network access."""
    report = build_supply_chain_evidence(output_root=tmp_path)

    assert report["status"] == "pass"
    assert set(report["generated_evidence"]) == EVIDENCE_IDS

    for evidence_id, filename in {
        "security_audit": "security-audit.json",
        "sbom": "sbom.json",
        "license_inventory": "license-inventory.json",
        "provenance": "provenance.json",
        "checksums": "checksums.json",
    }.items():
        artifact = tmp_path / filename
        assert artifact.is_file(), evidence_id
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["status"] == "pass"
        assert payload["evidence_id"] == evidence_id
        assert payload["generated_by"] == "scripts/release_supply_chain.py"
        assert payload["requires_secrets"] is False


def test_supply_chain_evidence_populates_release_readiness_report(tmp_path: Path) -> None:
    """Generated evidence should be fresh enough for the readiness evaluator."""
    build_supply_chain_evidence(output_root=tmp_path)

    report = build_readiness_report(evidence_root=tmp_path)

    assert "security_audit" not in report["missing_evidence"]
    assert "sbom" not in report["missing_evidence"]
    assert "license_inventory" not in report["missing_evidence"]
    assert "provenance" not in report["missing_evidence"]
    assert "checksums" not in report["missing_evidence"]
    assert report["status_counts"]["pass"] >= len(EVIDENCE_IDS)


def test_supply_chain_cli_writes_committed_evidence_root(tmp_path: Path) -> None:
    """The CLI should support explicit output roots for CI and local use."""
    output_root = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/release_supply_chain.py",
            "--output-root",
            str(output_root),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in output_root.glob("*.json")) == [
        "checksums.json",
        "license-inventory.json",
        "provenance.json",
        "sbom.json",
        "security-audit.json",
    ]
    assert json.loads(result.stdout)["status"] == "pass"


def test_committed_supply_chain_evidence_is_present_after_generation() -> None:
    """The repository should carry current supply-chain evidence for docs and Conductor."""
    for filename in (
        "security-audit.json",
        "sbom.json",
        "license-inventory.json",
        "provenance.json",
        "checksums.json",
    ):
        assert (EVIDENCE_ROOT / filename).is_file()
