"""Tests for roadmap gap-track drafting artifacts."""

from __future__ import annotations

import json
from pathlib import Path

GAP_TRACKS = {
    "lifecourse_adoption_fixture_20260504": (
        "Lifecourse Adoption-Trajectory Fixture",
        ("lifecourse", "adoption-trajectory", "Arrow", "base dependency"),
    ),
    "voiage_uncertainty_fixture_20260504": (
        "Voiage Diffusion-Uncertainty Fixture",
        ("voiage", "diffusion-uncertainty", "VOI", "base dependency"),
    ),
    "operational_modeling_fixtures_20260504": (
        "Operational Modeling Fixture Contracts",
        ("TreeAge-style", "DES", "event logs", "XLA"),
    ),
    "heoml_schema_placement_20260504": (
        "HEOML Schema Placement Decision",
        ("HEOML", "schema", "versioning", "migration"),
    ),
    "mars_surrogate_benchmark_gate_20260504": (
        "MARS Surrogate Benchmark Gate",
        ("mars", "benchmark", "JAX/XLA", "optional"),
    ),
}


def test_roadmap_gap_tracks_have_complete_conductor_artifacts() -> None:
    """Completed roadmap gaps should have archived Conductor artifacts."""
    for track_id, (title, _) in GAP_TRACKS.items():
        track_dir = Path("conductor/archive") / track_id
        metadata = json.loads((track_dir / "metadata.json").read_text())

        assert (track_dir / "spec.md").is_file(), track_id
        assert (track_dir / "plan.md").is_file(), track_id
        assert (track_dir / "index.md").is_file(), track_id
        assert metadata["track_id"] == track_id
        assert metadata["type"] == "chore"
        assert metadata["status"] == "completed"
        assert title in (track_dir / "spec.md").read_text()


def test_roadmap_gap_tracks_reference_confirmed_gap_sources() -> None:
    """Each drafted gap should trace back to roadmap or ecosystem source docs."""
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()
    contract = Path("specs/ecosystem/README.md").read_text()

    for track_id, (_, required_terms) in GAP_TRACKS.items():
        spec = (Path("conductor/archive") / track_id / "spec.md").read_text()
        plan = (Path("conductor/archive") / track_id / "plan.md").read_text()

        assert "Roadmap Source" in spec
        assert "docs/ecosystem/module_incubation_strategy.md" in spec
        assert "specs/ecosystem/README.md" in spec
        assert "Conductor - Automated Review and Checkpoint" in plan

        for term in required_terms:
            assert term in spec or term in plan, f"{term!r} missing from {track_id}"

    for source_term in (
        "adoption-trajectory fixture",
        "diffusion-uncertainty fixture",
        "TreeAge-style operational modeling fixture",
        "DES fixture",
        "HEOML extension schemas",
        "Benchmark whether `mars` improves",
    ):
        assert source_term in strategy

    for source_term in ("TreeAge-style", "DES adapters", "decision-analysis"):
        assert source_term in contract


def test_roadmap_gap_tracks_are_registered_and_resolvable() -> None:
    """Confirmed roadmap gaps should be registered as completed Conductor tracks."""
    registry = Path("conductor/tracks.md").read_text()

    for track_id, (title, _) in GAP_TRACKS.items():
        assert f"- [x] **Track: {title}** *(Completed)*" in registry
        assert f"./archive/{track_id}/" in registry
        assert (Path("conductor/archive") / track_id / "spec.md").is_file()


def test_roadmap_gap_tracks_keep_ecosystem_contracts_portable() -> None:
    """Gap tracks should preserve artifact-first, binding-friendly contracts."""
    for track_id in GAP_TRACKS:
        track_dir = Path("conductor/archive") / track_id
        spec = (track_dir / "spec.md").read_text()
        normalized_spec = spec.lower()

        assert "private" in normalized_spec or ("public" in normalized_spec and "api" in normalized_spec)
        assert "base" in normalized_spec


def test_roadmap_gap_tracks_have_common_validation_patterns() -> None:
    """Plans should converge on focused tests and explicit validation gates."""
    for track_id in GAP_TRACKS:
        plan = (Path("conductor/archive") / track_id / "plan.md").read_text()
        normalized_plan = plan.lower()

        assert "write failing" in normalized_plan, track_id
        assert "run focused" in normalized_plan, track_id
        assert "validat" in normalized_plan, track_id
        assert "Conductor - Automated Review and Checkpoint" in plan, track_id
