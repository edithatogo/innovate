"""Tests for binding publication planning and multi-language CI coverage."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ALIGNED_VERSION = "0.5.0"


def _csharp_properties() -> dict[str, str]:
    project = ElementTree.parse("bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj")
    return {child.tag: child.text for group in project.findall("PropertyGroup") for child in group if child.text}


def test_binding_package_names_follow_language_suffix_policy() -> None:
    """Binding package names should follow the language suffix policy where valid."""
    typescript_package = json.loads(Path("bindings/typescript/package.json").read_text())
    rust_manifest = tomllib.loads(Path("bindings/rust/Cargo.toml").read_text())
    r_description = Path("bindings/r/DESCRIPTION").read_text()
    julia_project = tomllib.loads(Path("bindings/julia/Project.toml").read_text())
    docs = Path("docs/source/binding_publication_ci.rst").read_text()

    assert typescript_package["name"] == "innovate.ts"
    assert rust_manifest["package"]["name"] == "innovate-rs"
    assert "Package: innovate.R" in r_description
    assert julia_project["name"] == "Innovate"
    assert "innovate.go" in docs
    assert "innovate.jl" in docs
    assert "innovate.rs" in docs
    assert _csharp_properties()["PackageId"] == "innovate.cs"


def test_binding_versions_are_aligned_with_python_package_version() -> None:
    """All binding package versions should match the primary Python release."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_json = json.loads(Path("bindings/typescript/package.json").read_text())
    cargo = tomllib.loads(Path("bindings/rust/Cargo.toml").read_text())
    julia = tomllib.loads(Path("bindings/julia/Project.toml").read_text())
    julia_manifest = tomllib.loads(Path("bindings/julia/Manifest.toml").read_text())
    r_description = Path("bindings/r/DESCRIPTION").read_text()

    assert pyproject["project"]["version"] == ALIGNED_VERSION
    assert package_json["version"] == ALIGNED_VERSION
    assert cargo["package"]["version"] == ALIGNED_VERSION
    assert f"Version: {ALIGNED_VERSION}" in r_description
    assert julia["version"] == ALIGNED_VERSION
    assert next(package["version"] for package in julia_manifest["deps"]["Innovate"]) == ALIGNED_VERSION
    assert _csharp_properties()["Version"] == ALIGNED_VERSION


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
    assert '<license type=\\"expression\\">Apache-2.0</license>' in workflow
    assert '<repository type=\\"git\\" url=\\"https://github.com/edithatogo/innovate\\"' in workflow
    assert "<tags>innovate health-economics decision-analysis kernel bindings</tags>" in workflow
    assert "contentFiles/any/any/innovate/kernel_bridge.py" in workflow
    assert "NPM_TOKEN" in workflow
    assert "CARGO_REGISTRY_TOKEN" in workflow
    assert "NUGET_API_KEY" in workflow


def test_r_publish_workflow_matches_source_package_tarball_name() -> None:
    """The R publish workflow should match the package source tarball name."""
    workflow = Path(".github/workflows/bindings-publish.yml").read_text()
    r_description = Path("bindings/r/DESCRIPTION").read_text()

    assert "Package: innovate.R" in r_description
    assert "innovate.R_*.tar.gz" in workflow
    assert "innovate_*.tar.gz" not in workflow
    assert "R CMD check --as-cran" in workflow


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
