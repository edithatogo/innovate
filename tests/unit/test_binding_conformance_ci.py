"""Tests for binding conformance CI gates and artifacts."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(".github/workflows/binding-conformance.yml")
DOC = Path("docs/astro-site/src/content/docs/maintainers/binding-conformance.md")


def test_binding_conformance_workflow_runs_shared_contract_tests() -> None:
    """A dedicated workflow should block drift in shared binding evidence."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Binding Conformance" in workflow
    assert "uv run pytest" in workflow
    for test_path in [
        "tests/unit/test_polyglot_binding_conformance.py",
        "tests/unit/test_polyglot_binding_golden_fixtures.py",
        "tests/unit/test_polyglot_binding_hardening.py",
    ]:
        assert test_path in workflow


def test_binding_conformance_workflow_uploads_evidence_artifacts() -> None:
    """CI should publish the evidence payloads maintainers need to inspect."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(r"actions/upload-artifact@[0-9a-f]{40}\b", workflow)
    for artifact in [
        "docs/source/_static/binding_conformance_inventory.json",
        "docs/source/_static/binding_golden_fixtures.json",
        "docs/source/_static/binding_hardening_evidence.json",
    ]:
        assert artifact in workflow


def test_binding_conformance_ci_has_documented_local_fallback() -> None:
    """Local fallback commands should be explicit for missing toolchains."""
    doc = DOC.read_text(encoding="utf-8")
    starlight_config = Path("docs/astro-site/starlight.config.mjs").read_text(encoding="utf-8")

    assert "Binding conformance CI" in doc
    assert "uv run pytest" in doc
    assert "language-native package checks" in doc
    assert "toolchain is unavailable" in doc
    assert "/maintainers/binding-conformance/" in starlight_config
