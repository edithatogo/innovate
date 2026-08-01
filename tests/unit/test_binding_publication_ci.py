"""Tests for binding publication planning and multi-language CI coverage."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from xml.etree import ElementTree

ALIGNED_VERSION = "0.5.0"
PUBLICATION_DOCS = (
    Path("docs/astro-site/src/content/docs/maintainers/publication.md"),
    Path("docs/astro-site/src/content/docs/latest/maintainers/publication.md"),
)


def _publication_docs() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PUBLICATION_DOCS)


def _csharp_properties() -> dict[str, str]:
    project = ElementTree.parse("bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj")
    return {child.tag: child.text for group in project.findall("PropertyGroup") for child in group if child.text}


def test_binding_package_names_follow_language_suffix_policy() -> None:
    """Binding package names should follow the language suffix policy where valid."""
    typescript_package = json.loads(Path("bindings/typescript/package.json").read_text())
    rust_manifest = tomllib.loads(Path("bindings/rust/Cargo.toml").read_text())
    r_description = Path("bindings/r/DESCRIPTION").read_text()
    julia_project = tomllib.loads(Path("bindings/julia/Project.toml").read_text())
    docs = _publication_docs()

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
    docs = _publication_docs()

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

    assert "registry_submission_receipts" in docs
    assert "maintainer-managed handoff states" in docs
    assert "blocker notes" not in docs


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
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo package",
        "npm run schema:check",
        "npm run typecheck",
        "npm test",
        "npm pack --dry-run",
        "go test ./...",
        "julia --project=bindings/julia",
        "Run Julia installed-package smoke",
        "Rscript bindings/r/tests/run.R",
        "R CMD check --as-cran --no-manual innovate.R_*.tar.gz",
        "dotnet test bindings/csharp/Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj",
        "dotnet pack bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj",
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


def test_python_release_workflows_validate_distribution_metadata() -> None:
    """Python package release gates should validate built artifacts."""
    ci = Path(".github/workflows/ci.yml").read_text()
    pypi = Path(".github/workflows/pypi-publish.yml").read_text()
    testpypi = Path(".github/workflows/testpypi-publish.yml").read_text()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert "twine>=5.1" in Path("pyproject.toml").read_text()
    assert "uv build" in ci
    assert "uv run twine check dist/*" in ci
    assert "uv build" in pypi
    assert "uv run twine check dist/*" in pypi
    assert "pypa/gh-action-pypi-publish@release/v1" in pypi
    assert "repository-url: https://test.pypi.org/legacy/" in testpypi
    assert pyproject["project"]["requires-python"] == ">=3.14"


def test_typescript_npm_metadata_and_ci_pack_gate() -> None:
    """The npm package should expose explicit metadata and pack gates."""
    package_json = json.loads(Path("bindings/typescript/package.json").read_text())
    ci = Path(".github/workflows/ci.yml").read_text()
    vitest_config = Path("bindings/typescript/vitest.config.ts").read_text()

    assert package_json["private"] is False
    assert package_json["license"] == "Apache-2.0"
    assert package_json["main"] == "./dist/index.js"
    assert package_json["types"] == "./dist/index.d.ts"
    assert package_json["exports"]["."]["import"] == "./dist/index.js"
    assert package_json["exports"]["."]["types"] == "./dist/index.d.ts"
    assert "files" in package_json
    assert "dist" in package_json["files"]
    assert "inst/python/kernel_bridge.py" in package_json["files"]
    assert package_json["engines"]["node"] == ">=26"
    assert package_json["devDependencies"]["typescript"].startswith("^6.")
    assert package_json["devDependencies"]["vitest"].startswith("^4.")
    assert package_json["devDependencies"]["@vitest/coverage-v8"].startswith("^4.")
    assert package_json["devDependencies"]["@types/node"].startswith("^26.")
    assert "node-version: ${{ matrix.node-version }}" in ci
    assert 'node-version: ["26"]' in ci
    assert "npm pack --dry-run" in ci
    assert "fileParallelism: false" in vitest_config
    assert "testTimeout: 120000" in vitest_config


def test_rust_crates_metadata_and_package_gates() -> None:
    """The crates.io package should declare metadata and package checks."""
    cargo = tomllib.loads(Path("bindings/rust/Cargo.toml").read_text())
    ci = Path(".github/workflows/ci.yml").read_text()
    publish = Path(".github/workflows/bindings-publish.yml").read_text()

    package = cargo["package"]
    assert package["license"] == "Apache-2.0"
    assert package["rust-version"] == "1.85"
    assert package["readme"] == "README.md"
    assert "api-bindings" in package["categories"]
    assert "kernel" in package["keywords"]
    assert "cargo clippy --all-targets --all-features -- -D warnings" in ci
    assert "cargo package" in ci
    assert "cargo package --list" in publish
    assert "--allow-dirty" not in publish
    assert "inst/python/kernel_bridge.py" in publish


def test_julia_registry_metadata_has_compat_bounds() -> None:
    """Julia General registry readiness requires dependency compat bounds."""
    julia = tomllib.loads(Path("bindings/julia/Project.toml").read_text())
    workflow = Path(".github/workflows/bindings-publish.yml").read_text()
    docs = _publication_docs()

    assert julia["name"] == "Innovate"
    assert julia["uuid"] == "ffe8f1e4-c541-43d5-9f32-550aacc4f51a"
    assert julia["compat"]["julia"] == "1.12"
    assert julia["compat"]["JSON"] == "0.21"
    assert "Validate Julia registry metadata" in workflow
    assert "Run Julia installed-package smoke" in workflow
    assert "installed-package smoke validation" in docs


def test_julia_installed_package_smoke_gate_runs_copied_package_bridge() -> None:
    """Julia publication should exercise the copied-package installed smoke path."""
    ci = Path(".github/workflows/ci.yml").read_text()
    publish = Path(".github/workflows/bindings-publish.yml").read_text()
    smoke = Path("bindings/julia/test/installed_package_smoke.jl").read_text()
    runtests = Path("bindings/julia/test/runtests.jl").read_text()

    for workflow in (ci, publish):
        assert "Run Julia installed-package smoke" in workflow
        assert 'INNOVATE_JULIA_RUN_BRIDGE_SMOKE: "true"' in workflow
        assert 'INNOVATE_JULIA_RUN_INSTALLED_PACKAGE_SMOKE: "true"' in workflow
        assert 'INNOVATE_PYTHON_COMMAND: "uv run --directory ${{ github.workspace }} python"' in workflow
        assert 'cp -R bindings/julia "$tmpdir/Innovate"' in workflow
        assert "Pkg.instantiate(); Pkg.test()" in workflow

    assert ci.index("Run Julia binding tests") < ci.index("Run Julia installed-package smoke")
    assert publish.index("Validate Julia registry metadata") < publish.index("Run Julia installed-package smoke")
    assert publish.index("Run Julia installed-package smoke") < publish.index("Julia General registry publication gate")
    assert 'INNOVATE_JULIA_RUN_INSTALLED_PACKAGE_SMOKE", "false") == "true"' in runtests
    assert 'include("installed_package_smoke.jl")' in runtests
    assert "kernel_repo_root_or_nothing() === nothing" in smoke
    assert "kernel_discover_models()" in smoke
    assert 'record["key"] == "bass"' in smoke


def test_go_module_release_gate_documents_submodule_tag_pattern() -> None:
    """Go release validation should protect the submodule tag convention."""
    workflow = Path(".github/workflows/bindings-publish.yml").read_text()
    go_mod = Path("bindings/go/go.mod").read_text()

    assert "module github.com/edithatogo/innovate/bindings/go" in go_mod
    assert "go 1.25.0" in go_mod
    assert "bindings/go/${GITHUB_REF_NAME}" in workflow
    assert "bindings/go/${GITHUB_REF_NAME}" in workflow
    assert "Validate Go module release tag pattern" in workflow


def test_csharp_nuget_ci_packs_every_target_framework() -> None:
    """NuGet readiness should include pack checks in the regular CI matrix."""
    ci = Path(".github/workflows/ci.yml").read_text()

    assert "dotnet pack bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj" in ci
    assert "bindings/csharp/artifacts/${{ matrix.target-framework }}" in ci
    assert "-p:TargetFrameworks=${{ matrix.target-framework }}" in ci


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
    docs = _publication_docs()

    assert '<None Include="../README.md" Pack="true" PackagePath="/" />' in project
    assert 'Include="../inst/python/kernel_bridge.py"' in project
    assert 'PackagePath="contentFiles/any/any/innovate/kernel_bridge.py"' in project
    assert 'PackageCopyToOutput="true"' in project
    assert "bridge-content" in docs
    assert "contentFiles/any/any/innovate/kernel_bridge.py" in docs
