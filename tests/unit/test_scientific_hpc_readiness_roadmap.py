"""Governance checks for the scientific and HPC readiness roadmap."""

from __future__ import annotations

import json
from pathlib import Path

ROADMAP = Path("docs/source/scientific_hpc_readiness_roadmap.rst")
TRACKS = Path("conductor/tracks.md")

FOLLOW_ON_TRACKS = {
    "Community Submission Readiness Matrix": "community_submission_readiness_20260507",
    "HPC Packaging and Registry Readiness": "hpc_packaging_registry_readiness_20260507",
    "Accelerator and Parallel Execution Evidence": "accelerator_parallel_execution_evidence_20260507",
    "Rust Core Migration Execution Plan": "rust_core_migration_execution_20260507",
    "ABI and Binary Compatibility Strategy": "abi_binary_compatibility_strategy_20260507",
    "Polyglot Repository and Documentation Architecture": "polyglot_docs_repo_architecture_20260507",
    "External Governance and Sustainability Dossier": "external_governance_sustainability_20260507",
}
COMPLETED_FOLLOW_ON_TRACKS = {
    "ABI and Binary Compatibility Strategy": "abi_binary_compatibility_strategy_20260507",
}


def test_scientific_hpc_readiness_roadmap_is_in_sphinx_navigation() -> None:
    """The strategic roadmap should be reachable from the docs site."""
    index = Path("docs/source/index.rst").read_text()

    assert ROADMAP.is_file()
    assert "scientific_hpc_readiness_roadmap" in index


def test_scientific_hpc_readiness_roadmap_covers_submission_targets() -> None:
    """The roadmap should cover the requested scientific community targets."""
    roadmap = ROADMAP.read_text()

    for target in (
        "Apache Arrow",
        "PyPA",
        "pyOpenSci",
        "rOpenSci",
        "JOSS",
        "NumFOCUS",
        "HPSF",
        "E4S",
        "Spack",
        "EasyBuild",
        "scikit-learn-contrib",
        ".NET Foundation",
        "Julia and R communities",
    ):
        assert target in roadmap


def test_scientific_hpc_readiness_roadmap_covers_hpc_and_abi_gaps() -> None:
    """HPC, accelerator, ABI, and API compatibility gaps should be explicit."""
    roadmap = ROADMAP.read_text()

    for phrase in (
        "CPU, GPU, TPU",
        "ASIC-oriented runtimes",
        "distributed execution",
        "scheduler-aware examples",
        "Slurm/PBS-style environments",
        "Arrow C Data Interface",
        "XLA, ``jaxlib``, Rust internal structs",
        "do not become public ABI",
        "public APIs stay semantic-versioned and schema-versioned",
    ):
        assert phrase in roadmap


def test_scientific_hpc_follow_on_tracks_are_registered_and_parallel_ready() -> None:
    """Follow-on tracks should exist with dependencies and parallel ownership."""
    registry = TRACKS.read_text()
    roadmap = ROADMAP.read_text()

    assert "Agent A" in roadmap
    assert "Agent F" in roadmap
    assert "Dependency graph" in roadmap

    for title, track_id in FOLLOW_ON_TRACKS.items():
        completed = title in COMPLETED_FOLLOW_ON_TRACKS
        track_dir = Path("conductor/archive" if completed else "conductor/tracks") / track_id
        metadata = json.loads((track_dir / "metadata.json").read_text())
        track_text = (track_dir / "spec.md").read_text() + (track_dir / "plan.md").read_text()

        if completed:
            assert f"- [x] **Track: {title}** *(Completed)*" in registry
            assert f"./archive/{track_id}/" in registry
        else:
            assert (
                f"- [ ] **Track: {title}**" in registry
                or f"- [~] **Track: {title}**" in registry
            )
            assert f"./tracks/{track_id}/" in registry
        assert title in roadmap
        assert metadata["track_id"] == track_id
        if completed:
            assert metadata["status"] == "completed"
        else:
            assert metadata["status"] in {"new", "in_progress"}
        assert "Dependencies" in track_text
        assert "Parallelization" in track_text
        assert (track_dir / "index.md").is_file()
