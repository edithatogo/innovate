"""Tests for Conductor workflow and tech-stack governance decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path


def test_python_task_runner_remains_uv_first() -> None:
    """The repo should keep uv primary while exposing nox orchestration."""
    tech_stack = Path("conductor/tech-stack.md").read_text()
    workflow = Path("conductor/workflow.md").read_text()

    assert "Python dependency management remains `uv`-first" in tech_stack
    assert "**nox** — Python task orchestration" in tech_stack
    assert "`uv` is the canonical Python dependency manager" in workflow
    assert "`nox` provides the Python task layer" in workflow
    assert "scripts/sync_versions.py" in tech_stack


def test_rust_profiling_stack_records_cpu_memory_and_gpu_scope() -> None:
    """The tech stack should record Rust profiling tools and GPU ownership."""
    tech_stack = Path("conductor/tech-stack.md").read_text()

    assert "**cargo-flamegraph**" in tech_stack
    assert "**DHAT**" in tech_stack
    assert "JAX/XLA device profilers" in tech_stack
    assert "until Rust owns a promoted native GPU execution backend" in tech_stack


def test_starlight_roadmap_track_records_versioned_plugin_baseline() -> None:
    """The Starlight track should pin the current docs package baseline."""
    tech_stack = Path("conductor/tech-stack.md").read_text()
    spec = Path("conductor/archive/starlight_versions_plugins_20260506/spec.md").read_text()
    plan = Path("conductor/archive/starlight_versions_plugins_20260506/plan.md").read_text()

    assert "@astrojs/starlight" in tech_stack
    assert "^0.41.3" in tech_stack
    assert "@astrojs/markdown-remark" in tech_stack
    assert "^7.2.0" in tech_stack
    assert "starlight-versions" in tech_stack
    assert "0.9.1" in tech_stack
    assert "starlight-links-validator" in tech_stack
    assert "0.25.2" in tech_stack
    assert "@astrojs/starlight-docsearch" in tech_stack
    assert "0.7.0" in tech_stack
    assert "@astrojs/sitemap" in tech_stack
    assert "Astro/Starlight Documentation Site Migration" in tech_stack
    assert "starlight-versions" in spec
    assert "starlight-links-validator" in spec
    assert "@astrojs/starlight-docsearch" in spec
    assert "Record whether DocSearch is selected or left as a future option" in plan


def test_future_astro_starlight_migration_track_records_cutover_gates() -> None:
    """The new Astro/Starlight migration track should encode the recommended gates."""
    spec = Path("conductor/archive/astro_starlight_docs_migration_20260511/spec.md").read_text()
    plan = Path("conductor/archive/astro_starlight_docs_migration_20260511/plan.md").read_text()

    for phrase in (
        "parallel-run or full cutover",
        "content inventory",
        "redirect inventory",
        "migration mode",
        "search provider",
        "@astrojs/sitemap",
        "route stability",
        "machine-readable migration manifest",
    ):
        assert phrase in spec or phrase in plan


def test_completed_archive_status_text_matches_metadata() -> None:
    """Completed Conductor archive indexes should not retain planned status text."""
    registry = Path("conductor/tracks.md").read_text()
    completed_archive_links = re.findall(
        r"- \[x\] \*\*Track: ([^*]+?)\*\*.*?\n\s+\*Link: \[\./archive/([^/]+)/\]",
        registry,
    )

    assert completed_archive_links
    for title, track_id in completed_archive_links:
        archive_dir = Path("conductor/archive") / track_id
        metadata = json.loads((archive_dir / "metadata.json").read_text())
        index_text = (archive_dir / "index.md").read_text()

        if metadata["status"] == "completed" and "## Status" in index_text:
            assert "Completed." in index_text, title
            assert "Planned." not in index_text, title
