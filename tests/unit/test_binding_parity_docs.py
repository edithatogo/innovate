"""Tests for binding parity documentation and stale claims."""

from __future__ import annotations

from pathlib import Path

CURRENT = Path("docs/astro-site/src/content/docs/bindings/parity.md")
LATEST = Path("docs/astro-site/src/content/docs/latest/bindings/parity.md")


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


def test_binding_docs_do_not_keep_stale_sphinx_cutover_claims() -> None:
    """Binding pages should not reference an obsolete Sphinx cutover."""
    stale_claims = (
        "until the Astro site fully replaces the Sphinx entry points",
        "Sphinx",
        "sphinx",
    )
    docs = [
        Path("docs/astro-site/src/content/docs/bindings/index.md"),
        Path("docs/astro-site/src/content/docs/latest/bindings/index.md"),
        CURRENT,
        LATEST,
    ]

    for path in docs:
        text = path.read_text(encoding="utf-8")
        for stale_claim in stale_claims:
            assert stale_claim not in text
