"""Tests for mapping roadmap items into Conductor records."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROADMAP_PATH = Path("docs/architecture_modernization_roadmap.md")

PRIMARY_ROADMAP_TRACKS = {
    "Canonical Public API and Package Topology": "../conductor/archive/canonical_api_topology_20260415/",
    "Optional Backends and Dependency Stabilization": "../conductor/archive/optional_backends_stabilization_20260415/",
    "Quality Gates and Release Hardening": "../conductor/archive/quality_gates_release_20260415/",
    "Functional Kernel Contract": "../conductor/archive/functional_kernel_contract_20260415/",
    "Arrow Interchange and Schema Layer": "../conductor/archive/arrow_interchange_schema_20260416/",
    "Plugin API and Stability Tiers": "../conductor/archive/plugin_api_stability_tiers_20260415/",
    "R Bindings over the Functional Kernel": "../conductor/archive/r_bindings_kernel_20260415/",
    "Julia Bindings over the Functional Kernel": "../conductor/archive/julia_bindings_kernel_20260415/",
    "TypeScript Bindings over the Functional Kernel": "../conductor/archive/typescript_bindings_kernel_20260416/",
    "Go Bindings over the Functional Kernel": "../conductor/archive/go_bindings_kernel_20260416/",
    "Rust Bindings over the Functional Kernel": "../conductor/archive/rust_bindings_kernel_20260416/",
    "Binding Publication and Multi-Language CI": "../conductor/archive/binding_publication_ci_20260428/",
    "Advanced Diffusion Inference": "../conductor/archive/advanced_diffusion_inference_20260415/",
    "Benchmark Corpus and Model Cards": "../conductor/archive/benchmark_corpus_modelcards_20260415/",
    "Rust Core Kernel Roadmap and C# Binding Foundation": "../conductor/archive/rust_core_kernel_20260428/",
}

ADR_RECORDS = {
    "ADR 0001: Array API and Arrow Foundation": "./adr/0001-array-api-and-arrow-foundation.md",
    "ADR 0002: JAX Is an Optional Accelerator Backend": "./adr/0002-jax-is-an-optional-accelerator-backend.md",
    "ADR 0003: Python DataFrame Strategy": "./adr/0003-python-dataframe-strategy.md",
    "ADR 0004: Core API, Bindings, and Rust Core Trajectory": "./adr/0004-core-api-bindings-and-rust-core-trajectory.md",
}

GOAL_PRINCIPLES = (
    "Array API for numerical portability",
    "Arrow for durable interchange",
    "JAX as an optional accelerator backend",
    "pandas plus PyArrow as the primary Python tabular surface",
    "selective, not foundational, use of Polars",
    "Python-first API stabilization followed by thin language bindings",
    "Rust Core Runtime as the strategic long-term execution direction",
)

COMPLETED_ROADMAP_FOLLOW_ON_TRACKS = {
    "wider probabilistic inference coverage": (
        "Probabilistic Inference Expansion",
        "probabilistic_inference_expansion_20260430",
    ),
    "richer diagnostics and uncertainty tooling": (
        "Diagnostics and Uncertainty Expansion",
        "diagnostics_uncertainty_expansion_20260430",
    ),
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

ROADMAP_GAP_TRACKS = {
    "Lifecourse Adoption-Trajectory Fixture": "lifecourse_adoption_fixture_20260504",
    "Voiage Diffusion-Uncertainty Fixture": "voiage_uncertainty_fixture_20260504",
    "Operational Modeling Fixture Contracts": "operational_modeling_fixtures_20260504",
    "HEOML Schema Placement Decision": "heoml_schema_placement_20260504",
    "MARS Surrogate Benchmark Gate": "mars_surrogate_benchmark_gate_20260504",
}

XLA_STRATEGY_TRACK = (
    "XLA Backend Strategy and JAX Kernel Promotion Gates",
    "xla_backend_strategy_20260430",
)


def test_completed_roadmap_follow_on_tracks_are_archived() -> None:
    """Completed follow-on tracks should be mapped to their Conductor archives."""
    roadmap = ROADMAP_PATH.read_text()
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
    roadmap = ROADMAP_PATH.read_text()
    registry = Path("conductor/tracks.md").read_text()
    title, track_id = ROADMAP_AUDIT_TRACK

    assert title in roadmap
    assert f"../conductor/archive/{track_id}/" in roadmap
    assert "implied work" in roadmap
    assert f"- [x] **Track: {title}** *(Completed)*" in registry
    assert f"./archive/{track_id}/" in registry


def test_roadmap_gap_tracks_are_registered() -> None:
    """Confirmed roadmap audit gaps should have completed Conductor records."""
    roadmap = ROADMAP_PATH.read_text()
    registry = Path("conductor/tracks.md").read_text()

    for title, track_id in ROADMAP_GAP_TRACKS.items():
        assert title in roadmap
        assert f"../conductor/archive/{track_id}/" in roadmap
        assert f"- [x] **Track: {title}** *(Completed)*" in registry
        assert f"./archive/{track_id}/" in registry
        assert (Path("conductor/archive") / track_id / "spec.md").is_file()


def test_active_roadmap_track_artifacts_exist() -> None:
    """Each completed roadmap gap track should have complete Conductor files."""
    completed_gap_tracks = [
        *ROADMAP_GAP_TRACKS.items(),
    ]
    for title, track_id in completed_gap_tracks:
        track_dir = Path("conductor/archive") / track_id
        metadata = json.loads((track_dir / "metadata.json").read_text())

        assert (track_dir / "spec.md").is_file(), title
        assert (track_dir / "plan.md").is_file(), title
        assert (track_dir / "index.md").is_file(), title
        assert metadata["track_id"] == track_id
        assert metadata["status"] == "completed"


def test_roadmap_primary_tracks_are_mapped_to_conductor_records() -> None:
    """Every primary roadmap track should have a resolving Conductor link."""
    roadmap = ROADMAP_PATH.read_text()

    for title, link in PRIMARY_ROADMAP_TRACKS.items():
        record_dir = (ROADMAP_PATH.parent / link).resolve()

        assert title in roadmap
        assert link in roadmap
        assert record_dir.is_dir(), title
        assert (record_dir / "spec.md").is_file(), title
        assert (record_dir / "plan.md").is_file(), title
        assert (record_dir / "index.md").is_file(), title


def test_roadmap_goal_principles_are_mapped() -> None:
    """The coverage map should account for each high-level roadmap principle."""
    roadmap = ROADMAP_PATH.read_text()

    assert "## Roadmap Coverage Map" in roadmap
    for principle in GOAL_PRINCIPLES:
        assert f"| {principle} |" in roadmap


def test_roadmap_adr_links_are_mapped_and_resolve() -> None:
    """Every ADR decision link should resolve and have Conductor coverage."""
    roadmap = ROADMAP_PATH.read_text()

    for title, link in ADR_RECORDS.items():
        record_path = (ROADMAP_PATH.parent / link).resolve()

        assert title in roadmap
        assert link in roadmap
        assert record_path.is_file(), title


def test_roadmap_links_resolve() -> None:
    """All active links in the roadmap should resolve from the roadmap file."""
    roadmap = ROADMAP_PATH.read_text()
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", roadmap)

    assert links
    for link in links:
        if link.startswith(("http://", "https://", "mailto:")):
            continue

        assert (ROADMAP_PATH.parent / link).resolve().exists(), link


def test_roadmap_status_language_separates_archive_from_active_backlog() -> None:
    """Status prose should not call archived follow-on tracks active."""
    roadmap = ROADMAP_PATH.read_text()
    normalized_roadmap = " ".join(roadmap.split())

    assert "Stage work, deferred follow-on tracks, and the ecosystem gap tracks" in normalized_roadmap
    assert "have been completed and archived" in normalized_roadmap
    assert "active backlog currently consists of" not in normalized_roadmap
    assert "ecosystem gap tracks registered by the audit" in normalized_roadmap
    assert "`Rust Core Expansion`, `C# Package Publication`, `Roadmap Completeness Audit`" not in normalized_roadmap
    assert "remaining strategic follow-ons" not in roadmap
    assert "are now active Conductor tracks" not in roadmap
    assert "converted them into Conductor records" in normalized_roadmap


def test_xla_strategy_is_registered_and_linked_from_roadmap() -> None:
    """XLA preference should be visible before follow-on tracks are implemented."""
    roadmap = ROADMAP_PATH.read_text()
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
