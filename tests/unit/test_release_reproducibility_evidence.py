"""Tests for release reproducibility evidence generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.release_readiness import build_readiness_report
from scripts.release_reproducibility import build_reproducibility_evidence

EVIDENCE_ROOT = Path("docs/source/_static/release_readiness/evidence")


def test_reproducibility_evidence_generator_writes_required_artifact(tmp_path: Path) -> None:
    """Reproducibility evidence should cover deterministic fixtures and seeded simulations."""
    report = build_reproducibility_evidence(output_root=tmp_path)

    assert report["status"] == "pass"
    assert report["generated_evidence"] == ["reproducibility"]

    artifact = tmp_path / "reproducibility.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["evidence_id"] == "reproducibility"
    assert payload["generated_by"] == "scripts/release_reproducibility.py"
    assert payload["seeded_simulation"]["seed"] == 42
    assert payload["seeded_simulation"]["first_digest"] == payload["seeded_simulation"]["second_digest"]
    assert payload["benchmark_fixture"]["first_digest"] == payload["benchmark_fixture"]["second_digest"]
    assert payload["generated_artifacts"]["readiness_contract_sha256"]
    assert payload["acceptable_nondeterminism"]
    assert payload["acceptable_nondeterminism"][0]["owner"]
    assert payload["acceptable_nondeterminism"][0]["rationale"]


def test_reproducibility_evidence_populates_release_readiness_report(tmp_path: Path) -> None:
    """Generated reproducibility evidence should satisfy the readiness contract item."""
    build_reproducibility_evidence(output_root=tmp_path)

    report = build_readiness_report(evidence_root=tmp_path)

    assert "reproducibility" not in report["missing_evidence"]
    assert any(check["id"] == "reproducibility" and check["status"] == "pass" for check in report["checks"])


def test_reproducibility_cli_writes_json_summary(tmp_path: Path) -> None:
    """The reproducibility generator should have a CI-friendly CLI."""
    output_root = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/release_reproducibility.py",
            "--output-root",
            str(output_root),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_root / "reproducibility.json").is_file()
    assert json.loads(result.stdout)["generated_evidence"] == ["reproducibility"]


def test_committed_reproducibility_evidence_is_present_after_generation() -> None:
    """The repository should carry current reproducibility evidence for release review."""
    payload = json.loads((EVIDENCE_ROOT / "reproducibility.json").read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["evidence_id"] == "reproducibility"
