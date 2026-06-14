"""Tests for the HPC submission packet."""

from __future__ import annotations

import json
from pathlib import Path

PACKET_PATH = Path("docs/source/_static/hpc_packaging/submission_packet.json")


def load_packet() -> dict[str, object]:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_hpc_submission_packet_exists() -> None:
    """The HPC packet should be stored as a durable machine-readable artifact."""
    assert PACKET_PATH.is_file()


def test_hpc_submission_packet_covers_all_targets() -> None:
    """The packet should enumerate every HPC registry target."""
    packet = load_packet()

    assert packet["schema_version"] == 1
    targets = {entry["target_id"]: entry for entry in packet["targets"]}
    assert set(targets) == {"spack", "easybuild", "hpsf", "e4s"}

    assert targets["spack"]["status"] == "ready_for_review"
    assert targets["easybuild"]["status"] == "ready_for_review"
    assert targets["hpsf"]["status"] == "blocked"
    assert targets["e4s"]["status"] == "blocked"

    for entry in targets.values():
        assert entry["registry"]
        assert entry["submission_mode"]
        assert entry["packet_artifacts"]
        assert entry["required_next_step"]


def test_hpc_submission_packet_points_at_existing_evidence() -> None:
    """The packet should reference evidence already present in the repo."""
    packet = load_packet()

    evidence_paths = {
        "docs/source/_static/hpc_packaging/evidence/python-install.log",
        "docs/source/_static/hpc_packaging/evidence/python-smoke.log",
        "docs/source/_static/hpc_packaging/evidence/r-build.log",
        "docs/source/_static/hpc_packaging/evidence/r-check.log",
        "docs/source/_static/hpc_packaging/evidence/rust-test.log",
        "docs/source/_static/hpc_packaging/evidence/julia-installed-smoke.log",
        "docs/source/_static/hpc_packaging/evidence/spack-batch.log",
        "docs/source/_static/hpc_packaging/evidence/easybuild-batch.log",
        "docs/source/_static/hpc_packaging/evidence/spack-pbs.log",
        "docs/source/_static/hpc_packaging/evidence/easybuild-pbs.log",
        "docs/source/_static/hpc_packaging/evidence/hpsf-review-note.md",
        "docs/source/_static/hpc_packaging/evidence/e4s-review-note.md",
        "docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log",
        "docs/source/_static/hpc_packaging/pack_packet.py",
        "docs/source/_static/hpc_packaging/workflow_manifest.json",
        "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json",
    }
    packet_artifacts = {artifact for entry in packet["targets"] for artifact in entry["packet_artifacts"]}

    assert evidence_paths.issubset(packet_artifacts)
