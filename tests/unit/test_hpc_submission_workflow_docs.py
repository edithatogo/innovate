"""Checks for the HPC submission workflow documentation."""

from __future__ import annotations

from pathlib import Path

DOC = Path("docs/source/hpc_submission_workflow.rst")
INDEX = Path("docs/source/index.rst")


def test_hpc_submission_workflow_doc_is_in_sphinx_navigation() -> None:
    assert DOC.is_file()
    index = INDEX.read_text(encoding="utf-8")
    assert "hpc_submission_workflow" in index


def test_hpc_submission_workflow_doc_mentions_all_targets() -> None:
    text = DOC.read_text(encoding="utf-8")

    for phrase in (
        "Spack",
        "EasyBuild",
        "HPSF",
        "E4S",
        "workflow_manifest.json",
        "hpc_submission_blockers.json",
        "hpc_submission_environment_probe.log",
        "Handoff status",
        "compatibility-named blocker bundle",
        "maintainer handoff states",
        "sbatch docs/source/_static/hpc_packaging/scheduler/slurm/spack-smoke.sbatch",
        "sbatch docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch",
    ):
        assert phrase in text

    assert "Any remaining blockers live in" not in text
    assert "review contact or blocker note" not in text
