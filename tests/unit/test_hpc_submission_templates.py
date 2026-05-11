"""Tests for the HPC submission templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path("docs/source/_static/hpc_packaging")


def test_hpc_submission_templates_exist() -> None:
    """The HPC packet should include executable scheduler and governance scaffolding."""
    for relative in (
        "scheduler/README.md",
        "scheduler/slurm/spack-smoke.sbatch",
        "scheduler/slurm/easybuild-smoke.sbatch",
        "scheduler/pbs/spack-smoke.pbs",
        "scheduler/pbs/easybuild-smoke.pbs",
        "governance/hpsf-evidence.md",
        "governance/e4s-evidence.md",
    ):
        assert (TEMPLATE_DIR / relative).is_file(), relative


def test_hpc_submission_templates_remain_non_claiming() -> None:
    """Templates should describe next steps instead of registry success."""
    readme = (TEMPLATE_DIR / "scheduler/README.md").read_text(encoding="utf-8")
    hpsf = (TEMPLATE_DIR / "governance/hpsf-evidence.md").read_text(encoding="utf-8")
    e4s = (TEMPLATE_DIR / "governance/e4s-evidence.md").read_text(encoding="utf-8")

    assert "cluster-job templates" in readme.lower()
    assert "not submission claims" in readme.lower() or "do not claim success" in readme.lower()
    assert "blocked until" in hpsf.lower()
    assert "blocked until" in e4s.lower()
