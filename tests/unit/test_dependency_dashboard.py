"""Tests for dependency-dashboard automation."""

from __future__ import annotations

from pathlib import Path

from scripts import dependency_dashboard


def test_dependency_dashboard_covers_required_ecosystems() -> None:
    """The dashboard should check every package ecosystem used by the repo."""
    check_ids = {check.check_id for check in dependency_dashboard.CHECKS}

    assert "uv-tree-outdated" in check_ids
    assert "pnpm-outdated-docs" in check_ids
    assert "npm-outdated-typescript-binding" in check_ids
    assert "cargo-update-dry-run" in check_ids
    assert "cargo-outdated" in check_ids
    assert "r-cran-outdated" in check_ids
    assert "julia-pkg-status-outdated" in check_ids
    assert "dotnet-outdated-runtime-package" in check_ids
    assert "dotnet-outdated-test-package" in check_ids


def test_dependency_dashboard_uses_non_mutating_commands() -> None:
    """Freshness commands should not mutate lockfiles or package manifests."""
    commands = {check.check_id: " ".join(check.command) for check in dependency_dashboard.CHECKS}

    assert "uv tree --outdated --frozen" in commands["uv-tree-outdated"]
    assert "pnpm outdated --format json" == commands["pnpm-outdated-docs"]
    assert "npm outdated --json" == commands["npm-outdated-typescript-binding"]
    assert "cargo update --dry-run" == commands["cargo-update-dry-run"]
    assert "cargo outdated" in commands["cargo-outdated"]
    assert "Pkg.status(; outdated=true)" in commands["julia-pkg-status-outdated"]
    assert "dotnet list" in commands["dotnet-outdated-runtime-package"]
    assert "--outdated" in commands["dotnet-outdated-test-package"]


def test_dependency_dashboard_workflow_runs_and_uploads_report() -> None:
    """GitHub Actions should schedule and upload the dependency dashboard."""
    workflow = Path(".github/workflows/dependency-dashboard.yml").read_text(encoding="utf-8")

    assert "name: Dependency Dashboard" in workflow
    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "uv run python scripts/dependency_dashboard.py" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "pnpm/action-setup@v4" in workflow
    assert "r-lib/actions/setup-r@v2" in workflow
    assert "julia-actions/setup-julia@v2" in workflow
    assert "actions/setup-dotnet@v5" in workflow
