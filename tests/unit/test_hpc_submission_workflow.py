"""Tests for the HPC submission workflow manifest."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path("docs/source/_static/hpc_packaging/workflow_manifest.json")
DOC_PATH = Path("docs/source/hpc_submission_workflow.rst")


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_hpc_submission_workflow_manifest_exists() -> None:
    assert MANIFEST_PATH.is_file()
    assert DOC_PATH.is_file()


def test_hpc_submission_workflow_manifest_covers_all_targets() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    targets = {entry["target_id"]: entry for entry in manifest["targets"]}
    assert set(targets) == {"spack", "easybuild", "hpsf", "e4s"}
    assert targets["spack"]["status"] == "ready_for_review"
    assert targets["easybuild"]["status"] == "ready_for_review"
    assert targets["hpsf"]["status"] == "blocked"
    assert targets["e4s"]["status"] == "blocked"
    for entry in targets.values():
        assert entry["commands"]
        assert entry["artifact_destinations"]
        assert entry["notes"]


def test_hpc_submission_workflow_commands_are_specific() -> None:
    manifest = load_manifest()
    targets = {entry["target_id"]: entry for entry in manifest["targets"]}

    assert "sbatch docs/source/_static/hpc_packaging/scheduler/slurm/spack-smoke.sbatch" in targets["spack"]["commands"]
    assert "sbatch docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch" in targets["easybuild"]["commands"]
    assert "edit docs/source/_static/hpc_packaging/governance/hpsf-evidence.md" in targets["hpsf"]["commands"]
    assert "edit docs/source/_static/hpc_packaging/governance/e4s-evidence.md" in targets["e4s"]["commands"]

    assert "docs/source/_static/hpc_packaging/evidence/spack-install.log" in targets["spack"]["artifact_destinations"]
    assert "docs/source/_static/hpc_packaging/evidence/easybuild-sanity.log" in targets["easybuild"]["artifact_destinations"]
    assert "docs/source/_static/hpc_packaging/evidence/spack-batch.log" in targets["spack"]["artifact_destinations"]
    assert "docs/source/_static/hpc_packaging/evidence/easybuild-batch.log" in targets["easybuild"]["artifact_destinations"]
    assert "docs/source/_static/hpc_packaging/evidence/hpsf-review-note.md" in targets["hpsf"]["artifact_destinations"]
    assert "docs/source/_static/hpc_packaging/evidence/e4s-review-note.md" in targets["e4s"]["artifact_destinations"]
    assert manifest["blocker_bundle"] == "evidence/hpc_submission_blockers.json"
