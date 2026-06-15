"""Tests for binding parity documentation and stale claims."""

from __future__ import annotations

from pathlib import Path

CURRENT = Path("docs/astro-site/src/content/docs/bindings/parity.md")
LATEST = Path("docs/astro-site/src/content/docs/latest/bindings/parity.md")
SPHINX = Path("docs/source/binding_parity.rst")
INDEX = Path("docs/source/index.rst")


def test_starlight_binding_parity_pages_cover_all_supported_languages() -> None:
    """Current and latest Starlight docs should expose binding parity evidence."""
    for page in [CURRENT, LATEST]:
        text = page.read_text(encoding="utf-8")
        assert "title: Binding parity" in text
        for language in ["Python", "Rust", "R", "Julia", "TypeScript", "Go", "C#"]:
            assert language in text
        for evidence in [
            "binding_conformance_inventory.json",
            "binding_golden_fixtures.json",
            "binding_hardening_evidence.json",
            "binding-conformance-evidence",
        ]:
            assert evidence in text


def test_sphinx_binding_parity_doc_links_evidence_and_package_receipts() -> None:
    """Legacy Sphinx docs should retain a parity entry during the cutover."""
    text = SPHINX.read_text(encoding="utf-8")

    assert "Binding parity" in text
    assert "registry_submission_receipts" in text
    assert "binding_conformance_ci" in text
    assert "language-native package checks" in text
    assert "binding_hardening_evidence.json" in text
    assert "binding_parity" in INDEX.read_text(encoding="utf-8")


def test_binding_docs_do_not_keep_stale_sphinx_cutover_claims() -> None:
    """Binding pages should not say Astro still needs to replace Sphinx."""
    stale_claim = "until the Astro site fully replaces the Sphinx entry points"
    docs = [
        Path("docs/astro-site/src/content/docs/bindings/index.md"),
        Path("docs/astro-site/src/content/docs/latest/bindings/index.md"),
        CURRENT,
        LATEST,
        SPHINX,
    ]

    for path in docs:
        assert stale_claim not in path.read_text(encoding="utf-8")
