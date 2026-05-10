"""Checks for the HPC registry contract page."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/source/hpc_registry_contract.rst")
INDEX_PATH = Path("docs/source/index.rst")


def test_hpc_registry_contract_is_in_primary_navigation() -> None:
    """The HPC contract should be reachable from the Sphinx landing page."""
    assert DOC_PATH.is_file()
    index = INDEX_PATH.read_text(encoding="utf-8")

    assert "hpc_registry_contract" in index


def test_hpc_registry_contract_describes_submission_boundary() -> None:
    """The contract should distinguish readiness from submission."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    for phrase in (
        "registry-facing contract",
        "not a submission claim",
        "Spack",
        "EasyBuild",
        "HPSF",
        "E4S",
        "scheduler-backed execution trace",
        "public kernel contract",
    ):
        assert phrase in docs


def test_hpc_registry_contract_lists_evidence_and_target_gates() -> None:
    """The document should list the evidence bundle and target gates."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    for phrase in (
        "Python wheel and sdist",
        "python -m pip check",
        "Rust test output",
        "Julia installed-package smoke evidence",
        "R build and ``R CMD check`` evidence",
        "Spack",
        "EasyBuild",
        "HPSF",
        "E4S",
    ):
        assert phrase in docs

