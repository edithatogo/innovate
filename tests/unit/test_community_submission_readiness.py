"""Governance checks for community submission readiness dossiers."""

from __future__ import annotations

import json
from pathlib import Path

MATRIX_PATH = Path("docs/source/_static/community_submission_readiness_matrix.json")
DOC_PATH = Path("docs/source/community_submission_readiness.rst")
INDEX_PATH = Path("docs/source/index.rst")

TARGETS = {
    "pyopensci",
    "ropensci",
    "joss",
    "numfocus",
    "scikit_learn_contrib",
    "apache_arrow",
    "dotnet_foundation",
    "julia_community",
    "r_community",
}

REQUIRED_EVIDENCE = {
    "docs",
    "tests",
    "examples",
    "citation",
    "governance",
    "maintenance",
}

VALID_STATUSES = {
    "ready",
    "near_ready",
    "blocked",
    "not_applicable",
}


def load_matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_community_submission_matrix_covers_requested_targets() -> None:
    """Every requested external community target should be represented."""
    matrix = load_matrix()
    targets = {target["id"] for target in matrix["targets"]}

    assert matrix["schema_version"] == 1
    assert targets == TARGETS


def test_community_submission_targets_have_status_evidence_and_blockers() -> None:
    """Readiness claims should keep evidence complete and blockers explicit when needed."""
    matrix = load_matrix()

    for target in matrix["targets"]:
        assert target["readiness_status"] in VALID_STATUSES
        assert set(target["reviewer_evidence"]) >= REQUIRED_EVIDENCE
        assert target["evidence_links"], target["id"]
        if target["readiness_status"] in {"ready", "not_applicable"}:
            assert target["blockers"] == [], target["id"]
        else:
            assert target["blockers"], target["id"]
            assert all(blocker["status"] in {"open", "blocked_external", "deferred"} for blocker in target["blockers"])


def test_community_submission_docs_link_matrix_and_sequence() -> None:
    """The reviewer-facing page should be navigable and link the matrix."""
    docs = DOC_PATH.read_text(encoding="utf-8")
    index = INDEX_PATH.read_text(encoding="utf-8")
    matrix = load_matrix()

    assert "community_submission_readiness" in index
    assert "community_submission_readiness_matrix.json" in docs
    assert "Submission sequencing" in docs
    assert "No submission claims readiness without evidence" in docs
    for target in matrix["targets"]:
        assert target["name"] in docs
