"""Guards for the polyglot repository and documentation architecture."""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("docs/source/polyglot_repo_architecture.rst")


def test_polyglot_architecture_page_is_in_primary_navigation() -> None:
    """The architecture guide should be reachable from the Sphinx landing page."""
    index = Path("docs/source/index.rst").read_text()

    assert DOC_PATH.exists()
    assert "polyglot_repo_architecture" in index
    assert index.index("polyglot_repo_architecture") < index.index("binding_publication_ci")


def test_polyglot_architecture_defines_audience_navigation_and_ownership() -> None:
    """The page should separate audiences and map repo ownership boundaries."""
    docs = DOC_PATH.read_text()

    for audience in (
        "Users",
        "Binding authors",
        "HPC administrators",
        "Maintainers",
    ):
        assert audience in docs

    for area in (
        "Core package",
        "Language bindings",
        "Packaging and release",
        "Scientific and HPC ecosystem",
        "Community submission dossiers",
    ):
        assert area in docs

    for binding in ("R", "Rust", "Julia", "TypeScript", "Go"):
        assert re.search(rf"\b{re.escape(binding)}\b", docs)
    assert "C# binding" in docs


def test_polyglot_architecture_records_layout_and_redirect_policy() -> None:
    """Layout guidance should preserve existing paths unless a redirect exists."""
    docs = DOC_PATH.read_text()

    assert "No source tree move is required for the current release" in docs
    assert "existing paths remain canonical" in docs.lower()
    assert "redirect" in docs.lower()

    for stable_path in (
        "docs/source/bindings.rst",
        "docs/source/binding_publication_ci.rst",
        "docs/source/scientific_hpc_readiness_roadmap.rst",
        "bindings/r/README.md",
        "bindings/rust/README.md",
        "bindings/julia/README.md",
        "bindings/typescript/README.md",
        "bindings/go/README.md",
        "bindings/csharp/README.md",
    ):
        assert stable_path in docs


def test_polyglot_architecture_proposes_target_navigation_without_source_moves() -> None:
    """The target architecture should be explicit enough for future tracks."""
    docs = DOC_PATH.read_text()

    assert "Target documentation architecture" in docs
    assert "Repository layout decision" in docs
    assert "docs-only reorganization" in docs
    assert "Source tree moves are deferred" in docs

    for target_section in (
        "Core contract",
        "Binding packages",
        "HPC deployment",
        "Submission evidence",
        "Maintainer decisions",
    ):
        assert target_section in docs


def test_polyglot_navigation_links_are_stable_and_bidirectional() -> None:
    """Navigation tests should prove new sections and old paths stay reachable."""
    bindings_hub = Path("docs/source/bindings.rst").read_text()
    architecture = DOC_PATH.read_text()

    assert "polyglot_repo_architecture" in bindings_hub
    assert "docs/source/polyglot_repo_architecture.rst" in architecture

    for stable_path in (
        "docs/source/bindings.rst",
        "docs/source/binding_publication_ci.rst",
        "docs/source/scientific_hpc_readiness_roadmap.rst",
        "docs/source/polyglot_repo_architecture.rst",
        "bindings/r/README.md",
        "bindings/rust/README.md",
        "bindings/julia/README.md",
        "bindings/typescript/README.md",
        "bindings/go/README.md",
        "bindings/csharp/README.md",
    ):
        assert Path(stable_path).exists(), stable_path
