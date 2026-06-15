"""Tests for maintainer-ready CRAN and scientific submission packets."""

from __future__ import annotations

import json
from pathlib import Path

PACKET_PATH = Path("docs/source/_static/scientific_submission_packet.json")
DOC_PATH = Path("docs/source/submission_readiness_dossiers.rst")

EXPECTED_TARGETS = {
    "r_cran",
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

VALID_STATES = {
    "ready_for_maintainer",
    "ready_for_review",
    "deferred",
    "not_applicable",
}


def load_packet() -> dict[str, object]:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_scientific_submission_packet_exists_and_covers_targets() -> None:
    """CRAN and scientific/community targets should have one packet artifact."""
    packet = load_packet()
    targets = {entry["target_id"]: entry for entry in packet["targets"]}

    assert packet["schema_version"] == 1
    assert packet["generated_by_track"] == "external_acceptance_completion_20260614"
    assert packet["packet_date"] == "2026-06-16"
    assert set(targets) == EXPECTED_TARGETS

    assert targets["r_cran"]["status"] == "ready_for_maintainer"
    assert targets["numfocus"]["status"] == "deferred"
    assert targets["scikit_learn_contrib"]["status"] == "not_applicable"


def test_scientific_submission_packets_are_owner_backed() -> None:
    """Every packet should have an owner, exact external path, and no submit claim."""
    packet = load_packet()

    for entry in packet["targets"]:
        assert entry["status"] in VALID_STATES, entry["target_id"]
        assert entry["owner"], entry["target_id"]
        assert entry["external_action_url"].startswith("https://"), entry["target_id"]
        assert entry["maintainer_action_boundary"], entry["target_id"]
        assert "maintainer" in entry["maintainer_action_boundary"].lower(), entry["target_id"]
        assert entry["receipt_rule"], entry["target_id"]
        assert entry["revisit_condition"], entry["target_id"]
        assert entry["packet_artifacts"], entry["target_id"]
        assert all("blocked" not in str(value).lower() for value in entry.values()), entry["target_id"]


def test_scientific_submission_packets_reference_existing_or_external_evidence() -> None:
    """Packet artifacts should be concrete local files/directories or URLs."""
    packet = load_packet()

    for entry in packet["targets"]:
        for artifact in entry["packet_artifacts"]:
            if artifact.startswith("https://"):
                continue
            assert Path(artifact).exists(), f"{entry['target_id']} references missing {artifact}"


def test_submission_dossiers_document_scientific_packet() -> None:
    """Reviewer-facing docs should link the machine-readable packet."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "scientific_submission_packet.json" in docs
    assert "CRAN and scientific submission packet" in docs
