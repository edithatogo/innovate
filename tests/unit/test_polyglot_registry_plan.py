"""Checks for the polyglot registry plan page."""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/astro-site/src/content/docs/operations/polyglot-registry.md")
LATEST_DOC_PATH = Path("docs/astro-site/src/content/docs/latest/operations/polyglot-registry.md")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")


def _docs_text() -> str:
    return "\n".join((DOC_PATH.read_text(encoding="utf-8"), LATEST_DOC_PATH.read_text(encoding="utf-8")))


def test_polyglot_registry_plan_is_in_primary_navigation() -> None:
    """The registry plan should be reachable from the Starlight sidebar."""
    assert DOC_PATH.is_file()
    assert LATEST_DOC_PATH.is_file()
    starlight_config = STARLIGHT_CONFIG.read_text(encoding="utf-8")

    assert "/operations/polyglot-registry/" in starlight_config
    assert "slug: latest/operations/polyglot-registry" in LATEST_DOC_PATH.read_text(encoding="utf-8")


def test_polyglot_registry_plan_distinguishes_registry_layers() -> None:
    """The document should separate package, scientific, and HPC registries."""
    docs = _docs_text()

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
    docs = _docs_text()

    for phrase in (
        "does not claim that any external submission has already been completed",
        "release decision, not a doc-only milestone",
        "Use the readiness dossiers as submission checklists, not as proof of submission",
        "Use the HPC registry contract",
    ):
        assert phrase in docs
