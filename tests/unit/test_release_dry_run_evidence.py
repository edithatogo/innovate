"""Tests for release dry-run evidence and documentation."""

from __future__ import annotations

import json
from pathlib import Path

DRY_RUN = Path("docs/source/_static/release_readiness/release-dry-run.json")
READINESS_DOC = Path("docs/source/release_readiness.rst")
STARLIGHT_DOC = Path("docs/astro-site/src/content/docs/maintainers/release-readiness.md")
STARLIGHT_LATEST_DOC = Path("docs/astro-site/src/content/docs/latest/maintainers/release-readiness.md")


def test_release_dry_run_records_all_package_surfaces() -> None:
    """The dry-run artifact should cover every package surface."""
    payload = json.loads(DRY_RUN.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["status"] == "release_candidate"
    assert payload["readiness_report"] == "docs/source/_static/release_readiness/readiness-report.json"
    assert {surface["id"] for surface in payload["package_surfaces"]} == {
        "python",
        "rust",
        "typescript",
        "r",
        "julia",
        "go",
        "csharp",
        "docs",
        "hpc",
    }
    for surface in payload["package_surfaces"]:
        assert surface["dry_run_command"]
        assert surface["status"] in {"passed", "documented", "blocked"}
        assert surface["evidence"]


def test_release_dry_run_links_registry_docs_to_readiness_report() -> None:
    """Registry and docs receipts should consume the release-readiness artifact."""
    payload = json.loads(DRY_RUN.read_text(encoding="utf-8"))

    assert "docs/source/registry_submission_receipts.rst" in payload["consumers"]
    assert "docs/source/hpc_packaging_registry_readiness.rst" in payload["consumers"]
    assert "docs/source/release_readiness.rst" in payload["consumers"]
    assert payload["final_gate_sequence"] == [
        "generate_supply_chain_evidence",
        "generate_reproducibility_evidence",
        "generate_release_readiness_report",
        "run_package_dry_runs",
        "review_release_candidate_blockers",
        "maintainer_approval_before_public_release",
    ]


def test_release_readiness_docs_include_final_gate_sequence() -> None:
    """Maintainer docs should explain the final release gate sequence."""
    for doc_path in (READINESS_DOC, STARLIGHT_DOC, STARLIGHT_LATEST_DOC):
        content = doc_path.read_text(encoding="utf-8")
        assert "Final gate sequence" in content
        assert "release-dry-run.json" in content
        assert "maintainer approval" in content.lower()
        assert "uv run nox -s package" in content
