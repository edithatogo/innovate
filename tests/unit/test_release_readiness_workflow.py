"""Tests for the GitHub Actions release-readiness workflow."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/release-readiness.yml")


def test_release_readiness_workflow_exists_and_runs_local_gates() -> None:
    """GitHub Actions should expose the same release-readiness command as local maintainers."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Release Readiness" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "uv run nox -s release_supply_chain" in workflow
    assert "uv run nox -s release_reproducibility" in workflow
    assert "uv run nox -s release_readiness" in workflow


def test_release_readiness_workflow_uploads_artifacts_and_splits_slow_lanes() -> None:
    """The workflow should upload reports and keep slow checks in explicit release lanes."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/upload-artifact" in workflow
    assert "release-readiness-report" in workflow
    assert "docs/source/_static/release_readiness/readiness-report.json" in workflow
    assert "release-lane-checks" in workflow
    assert "mutation sampling" in workflow.lower()
    assert "scheduled or manual release lane" in workflow.lower()
