"""Checks for the polyglot registry plan page."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/source/polyglot_registry_plan.rst")
INDEX_PATH = Path("docs/source/index.rst")


def test_polyglot_registry_plan_is_in_primary_navigation() -> None:
    """The registry plan should be reachable from the Sphinx landing page."""
    assert DOC_PATH.is_file()
    index = INDEX_PATH.read_text(encoding="utf-8")

    assert "polyglot_registry_plan" in index


def test_polyglot_registry_plan_distinguishes_registry_layers() -> None:
    """The document should separate package, scientific, and HPC registries."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    for phrase in (
        "Package-manager registries",
        "Scientific community submissions",
        "HPC registries",
        "PyPI/TestPyPI",
        "npm",
        "crates.io",
        "Julia General",
        "Go modules",
        "NuGet",
        "Spack",
        "EasyBuild",
    ):
        assert phrase in docs


def test_polyglot_registry_plan_states_it_is_not_a_submission_claim() -> None:
    """The plan should remain a planning artifact rather than a registry claim."""
    docs = DOC_PATH.read_text(encoding="utf-8")

    for phrase in (
        "does not claim that any external submission has already been completed",
        "release decision, not a doc-only milestone",
        "Use the readiness dossiers as submission checklists, not as proof of submission",
        "Use the HPC registry contract",
    ):
        assert phrase in docs
