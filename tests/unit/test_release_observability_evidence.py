"""Phase 3 observability and release evidence validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "docs/source/_static/release_readiness/evidence"
GATE_INVENTORY = ROOT / "conductor/archive/ci_code_quality_release_hardening_20260625/gate-inventory.md"
REQUIRED_FIELDS = {"evidence_id", "schema_version", "status", "generated_at"}
ARTIFACT_MAP = {
    "sbom": "sbom.json",
    "provenance": "provenance.json",
    "checksums": "checksums.json",
    "security_audit": "security-audit.json",
    "reproducibility": "reproducibility.json",
    "license_inventory": "license-inventory.json",
}
IDS = sorted(ARTIFACT_MAP)


def _load(eid: str) -> dict:
    p = EVIDENCE_ROOT / ARTIFACT_MAP[eid]
    assert p.is_file(), f"Missing: {p}"
    return json.loads(p.read_text())


class TestObservabilityEvidence:
    @pytest.mark.parametrize("eid", IDS)
    def test_artifact_exists(self, eid: str) -> None:
        assert (EVIDENCE_ROOT / ARTIFACT_MAP[eid]).is_file()

    @pytest.mark.parametrize("eid", IDS)
    def test_has_required_fields(self, eid: str) -> None:
        data = _load(eid)
        for f in REQUIRED_FIELDS:
            assert f in data, f"{eid} missing {f}"

    @pytest.mark.parametrize("eid", IDS)
    def test_evidence_id_matches(self, eid: str) -> None:
        assert _load(eid).get("evidence_id") == eid

    @pytest.mark.parametrize("eid", IDS)
    def test_valid_status(self, eid: str) -> None:
        valid = {"pass", "fail", "blocked", "deferred", "manual", "missing"}
        assert _load(eid).get("status") in valid


class TestNoxObservabilitySessions:
    def test_release_supply_chain_session(self) -> None:
        t = (ROOT / "noxfile.py").read_text()
        assert "def release_supply_chain" in t
        assert "scripts/release_supply_chain.py" in t

    def test_release_reproducibility_session(self) -> None:
        t = (ROOT / "noxfile.py").read_text()
        assert "def release_reproducibility" in t
        assert "scripts/release_reproducibility.py" in t

    def test_release_readiness_session(self) -> None:
        t = (ROOT / "noxfile.py").read_text()
        assert "def release_readiness" in t
        assert "scripts/release_readiness.py" in t


class TestStructuredLogging:
    def test_runtime_logging_docs_exist(self) -> None:
        p = ROOT / "docs/astro-site/src/content/docs/operations/runtime-logging.md"
        assert p.is_file(), "Missing runtime-logging.md"

    def test_runtime_logging_docs_mention_structured(self) -> None:
        p = ROOT / "docs/astro-site/src/content/docs/operations/runtime-logging.md"
        assert "structured" in p.read_text().lower()


class TestGateInventoryObservability:
    def test_lists_sbom(self) -> None:
        assert "sbom" in GATE_INVENTORY.read_text().lower()

    def test_lists_provenance(self) -> None:
        assert "provenance" in GATE_INVENTORY.read_text().lower()

    def test_lists_checksums(self) -> None:
        assert "checksum" in GATE_INVENTORY.read_text().lower()

    def test_lists_security_audit(self) -> None:
        t = GATE_INVENTORY.read_text().lower()
        assert "security" in t
        assert "audit" in t


class TestReleaseReadinessFreshness:
    def _report(self) -> dict:
        p = ROOT / "docs/source/_static/release_readiness/readiness-report.json"
        assert p.is_file(), "Missing readiness-report.json"
        return json.loads(p.read_text())

    def test_report_exists(self) -> None:
        self._report()

    def test_has_overall_status(self) -> None:
        assert "overall_status" in self._report()

    def test_has_release_state(self) -> None:
        assert "release_state" in self._report()

    def test_has_checks(self) -> None:
        d = self._report()
        assert "checks" in d
        assert len(d["checks"]) > 0

    def test_script_enforces_fail_closed(self) -> None:
        t = (ROOT / "scripts/release_readiness.py").read_text()
        assert "release_ready" in t
        assert "blocked" in t

    def test_contract_has_freshness_days(self) -> None:
        c = json.loads((ROOT / "docs/source/_static/release_readiness_contract.json").read_text())
        for e in c["required_evidence"]:
            assert e.get("freshness_days", 0) > 0, f"{e['id']} missing freshness"
