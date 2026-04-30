"""Tests for mapping roadmap deferred work into active Conductor tracks."""

from __future__ import annotations

import json
from pathlib import Path

COMPLETED_ROADMAP_FOLLOW_ON_TRACKS = {
    "wider probabilistic inference coverage": (
        "Probabilistic Inference Expansion",
        "probabilistic_inference_expansion_20260430",
    ),
    "richer diagnostics and uncertainty tooling": (
        "Diagnostics and Uncertainty Expansion",
        "diagnostics_uncertainty_expansion_20260430",
    ),
}

ROADMAP_BACKLOG_TRACKS = {
    "broader benchmark corpus automation": (
        "Benchmark Corpus Automation",
        "benchmark_corpus_automation_20260430",
    ),
    "hosted services or remote execution layers": (
        "Hosted Services and Remote Execution",
        "hosted_remote_execution_20260430",
    ),
    "aggressive DataFrame engine experimentation beyond ingestion and ETL edges": (
        "DataFrame Engine Experimentation",
        "dataframe_engine_experimentation_20260430",
    ),
    "broad Rust rewrites before operation-level parity and benchmark gates exist": (
        "Rust Core Expansion",
        "rust_core_expansion_20260430",
    ),
    "C# package publication before the thin-binding contract is validated": (
        "C# Package Publication",
        "csharp_package_publication_20260430",
    ),
}

ROADMAP_AUDIT_TRACK = (
    "Roadmap Completeness Audit",
    "roadmap_completeness_audit_20260430",
)

XLA_STRATEGY_TRACK = (
    "XLA Backend Strategy and JAX Kernel Promotion Gates",
    "xla_backend_strategy_20260430",
)


def test_deferred_roadmap_items_are_mapped_to_active_tracks() -> None:
    """Every explicit deferred roadmap item should point to an active track."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()
    registry = Path("conductor/tracks.md").read_text()

    for deferred_item, (title, track_id) in ROADMAP_BACKLOG_TRACKS.items():
        assert deferred_item in roadmap
        assert title in roadmap
        assert f"../conductor/tracks/{track_id}/" in roadmap
        assert f"- [ ] **Track: {title}**" in registry
        assert f"./tracks/{track_id}/" in registry


def test_completed_roadmap_follow_on_tracks_are_archived() -> None:
    """Completed follow-on tracks should be mapped to their Conductor archives."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()
    registry = Path("conductor/tracks.md").read_text()

    for deferred_item, (title, track_id) in COMPLETED_ROADMAP_FOLLOW_ON_TRACKS.items():
        track_dir = Path("conductor/archive") / track_id
        metadata = json.loads((track_dir / "metadata.json").read_text())

        assert deferred_item in roadmap
        assert title in roadmap
        assert f"../conductor/archive/{track_id}/" in roadmap
        assert f"- [x] **Track: {title}** *(Completed)*" in registry
        assert f"./archive/{track_id}/" in registry
        assert (track_dir / "spec.md").is_file()
        assert (track_dir / "plan.md").is_file()
        assert (track_dir / "index.md").is_file()
        assert metadata["track_id"] == track_id
        assert metadata["status"] == "completed"


def test_roadmap_completeness_audit_track_is_registered() -> None:
    """The roadmap should include a track for finding missing implied work."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()
    registry = Path("conductor/tracks.md").read_text()
    title, track_id = ROADMAP_AUDIT_TRACK

    assert title in roadmap
    assert f"../conductor/tracks/{track_id}/" in roadmap
    assert "implied work" in roadmap
    assert f"- [ ] **Track: {title}**" in registry
    assert f"./tracks/{track_id}/" in registry


def test_active_roadmap_track_artifacts_exist() -> None:
    """Each active roadmap backlog track should have complete Conductor files."""
    for title, track_id in [*ROADMAP_BACKLOG_TRACKS.values(), ROADMAP_AUDIT_TRACK]:
        track_dir = Path("conductor/tracks") / track_id
        metadata = json.loads((track_dir / "metadata.json").read_text())

        assert (track_dir / "spec.md").is_file(), title
        assert (track_dir / "plan.md").is_file(), title
        assert (track_dir / "index.md").is_file(), title
        assert metadata["track_id"] == track_id
        assert metadata["status"] == "new"


def test_xla_strategy_is_registered_and_linked_from_roadmap() -> None:
    """XLA preference should be visible before follow-on tracks are implemented."""
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()
    registry = Path("conductor/tracks.md").read_text()
    tech_stack = Path("conductor/tech-stack.md").read_text()
    title, track_id = XLA_STRATEGY_TRACK
    track_dir = Path("conductor/archive") / track_id
    metadata = json.loads((track_dir / "metadata.json").read_text())

    assert f"- [x] **Track: {title}** *(Completed)*" in registry
    assert f"./archive/{track_id}/" in registry
    assert f"../conductor/archive/{track_id}/" in roadmap
    assert (track_dir / "spec.md").is_file()
    assert (track_dir / "plan.md").is_file()
    assert (track_dir / "index.md").is_file()
    assert metadata["track_id"] == track_id
    assert metadata["status"] == "completed"
    assert "Prefer XLA-backed libraries" in roadmap

    for library in (
        "JAX",
        "NumPyro",
        "BlackJAX",
        "TensorFlow Probability",
        "Diffrax",
    ):
        assert library in roadmap
        assert library in tech_stack


def test_xla_strategy_is_reflected_in_dependent_tracks() -> None:
    """Dependent backlog tracks should evaluate XLA before non-XLA acceleration."""
    required_phrases_by_track = {
        "probabilistic_inference_expansion_20260430": (
            "NumPyro",
            "BlackJAX",
            "XLA eligibility",
        ),
        "diagnostics_uncertainty_expansion_20260430": (
            "JAX/XLA-backed diagnostics",
            "XLA eligibility",
        ),
        "benchmark_corpus_automation_20260430": (
            "XLA compilation cost",
            "steady-state runtime",
        ),
        "dataframe_engine_experimentation_20260430": (
            "XLA-backed numerical kernels",
            "tabular execution",
        ),
        "rust_core_expansion_20260430": (
            "JAX/XLA-backed implementations",
            "XLA compile cost",
        ),
        "hosted_remote_execution_20260430": (
            "JAX/XLA",
            "backend provenance",
        ),
    }

    for track_id, phrases in required_phrases_by_track.items():
        track_dir = Path("conductor/tracks") / track_id
        if not track_dir.exists():
            track_dir = Path("conductor/archive") / track_id
        track_text = (track_dir / "spec.md").read_text() + (track_dir / "plan.md").read_text()

        for phrase in phrases:
            assert phrase in track_text, f"{phrase!r} missing from {track_id}"
