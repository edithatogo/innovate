"""Tests for binding publication planning and multi-language CI coverage."""

from __future__ import annotations

from pathlib import Path


def test_binding_publication_docs_name_registry_targets() -> None:
    """Every planned binding should have a package-manager publication target."""
    docs = Path("docs/source/binding_publication_ci.rst").read_text()

    for target in (
        "npm",
        "crates.io",
        "R-universe",
        "CRAN",
        "Julia General",
        "Go modules",
        "NuGet",
    ):
        assert target in docs


def test_ci_workflow_runs_implemented_language_bindings() -> None:
    """The main CI workflow should validate every implemented binding toolchain."""
    workflow = Path(".github/workflows/ci.yml").read_text()

    for job_name in (
        "rust-bindings",
        "typescript-bindings",
        "go-bindings",
        "julia-bindings",
        "r-bindings",
    ):
        assert job_name in workflow

    for command in (
        "cargo test",
        "npm run schema:check",
        "npm run typecheck",
        "npm test",
        "go test ./...",
        "julia --project=bindings/julia",
        "Rscript bindings/r/tests/run.R",
    ):
        assert command in workflow


def test_binding_publish_workflow_has_release_gated_registry_steps() -> None:
    """Publication hooks should be explicit and gated on release/manual intent."""
    workflow = Path(".github/workflows/bindings-publish.yml").read_text()

    for registry_step in (
        "Publish to npm",
        "Publish to crates.io",
        "R CMD build bindings/r",
        "Julia General registry",
        "go list -m",
        "NuGet",
    ):
        assert registry_step in workflow

    assert "release:" in workflow
    assert "workflow_dispatch:" in workflow
    assert 'dotnet-version: "11.0.x"' in workflow
    assert "NPM_TOKEN" in workflow
    assert "CARGO_REGISTRY_TOKEN" in workflow
