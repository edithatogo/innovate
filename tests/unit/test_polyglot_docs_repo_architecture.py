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
