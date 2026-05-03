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
        "csharp-bindings",
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
        "dotnet test bindings/csharp/Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj",
        'dotnet-version: "10.0.x"',
        'dotnet-version: "11.0.x"',
        'target-framework: "net10.0"',
        'target-framework: "net11.0"',
        "-p:TargetFrameworks=${{ matrix.target-framework }}",
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
        "Tag Go submodule release",
        "Publish to NuGet",
        "dotnet pack bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj",
        "Validate NuGet package artifacts",
    ):
        assert registry_step in workflow

    assert "release:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "10.0.x" in workflow
    assert "11.0.x" in workflow
    assert "--framework net10.0 -p:TargetFrameworks=net10.0" in workflow
    assert "--framework net11.0 -p:TargetFrameworks=net11.0" in workflow
    assert "-p:ContinuousIntegrationBuild=true" in workflow
    assert "bindings/csharp/artifacts/*.nupkg" in workflow
    assert "bindings/csharp/artifacts/*.snupkg" in workflow
    assert '<license type=\\"expression\\">MIT</license>' in workflow
    assert '<repository type=\\"git\\" url=\\"https://github.com/edithatogo/innovate\\"' in workflow
    assert "<tags>innovate health-economics decision-analysis kernel bindings</tags>" in workflow
    assert "contentFiles/any/any/innovate/kernel_bridge.py" in workflow
    assert "NPM_TOKEN" in workflow
    assert "CARGO_REGISTRY_TOKEN" in workflow
    assert "NUGET_API_KEY" in workflow


def test_csharp_nuget_pack_includes_runtime_bridge_asset() -> None:
    """The NuGet package metadata should include assets needed by the thin bridge."""
    project = Path("bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj").read_text()
    docs = Path("docs/source/binding_publication_ci.rst").read_text()

    assert '<None Include="../README.md" Pack="true" PackagePath="/" />' in project
    assert 'Include="../inst/python/kernel_bridge.py"' in project
    assert 'PackagePath="contentFiles/any/any/innovate/kernel_bridge.py"' in project
    assert 'PackageCopyToOutput="true"' in project
    assert "bridge-content" in docs
    assert "contentFiles/any/any/innovate/kernel_bridge.py" in docs
