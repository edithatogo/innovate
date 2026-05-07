"""Governance checks for the external governance and sustainability dossier."""

from __future__ import annotations

import json
from pathlib import Path


DOC_PATH = Path("docs/source/external_governance_sustainability.rst")
MATRIX_PATH = Path("docs/source/_static/external_governance_sustainability_matrix.json")
INDEX_PATH = Path("docs/source/index.rst")


def load_matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_external_governance_dossier_is_in_sphinx_navigation() -> None:
    """The governance dossier should be reachable from the Sphinx landing page."""
    index = INDEX_PATH.read_text(encoding="utf-8")

    assert DOC_PATH.is_file()
    assert "external_governance_sustainability" in index


def test_external_governance_matrix_covers_core_governance_items() -> None:
    """The machine-readable matrix should enumerate the expected governance areas."""
    matrix = load_matrix()
    ids = {item["id"] for item in matrix["governance_items"]}

    assert matrix["schema_version"] == 1
    assert ids == {
        "maintainer_roles",
        "security_policy",
        "citation_metadata",
        "contributor_onboarding",
        "support_policy",
        "funding_path",
        "roadmap_ownership",
    }


def test_external_governance_dossier_links_existing_policy_files() -> None:
    """The dossier should point at the repo's current stewardship evidence."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    for phrase in (
        "CODEOWNERS",
        "SECURITY.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "scientific_hpc_readiness_roadmap",
        "architecture_modernization_roadmap",
        "support matrix",
        "funding or sponsorship statement",
    ):
        assert phrase in docs

