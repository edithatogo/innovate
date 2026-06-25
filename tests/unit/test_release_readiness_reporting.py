"""Tests for release-readiness report generation and maintainer docs."""

from __future__ import annotations

from pathlib import Path

NOXFILE = Path("noxfile.py")
STARLIGHT_DOC = Path("docs/astro-site/src/content/docs/maintainers/release-readiness.md")
STARLIGHT_LATEST_DOC = Path("docs/astro-site/src/content/docs/latest/maintainers/release-readiness.md")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")


def test_nox_exposes_release_readiness_session() -> None:
    """Maintainers should have one local command for release-readiness reports."""
    noxfile = NOXFILE.read_text(encoding="utf-8")

    assert "def release_readiness" in noxfile
    assert "scripts/release_readiness.py" in noxfile
    assert "--output" in noxfile
    assert "docs/source/_static/release_readiness/readiness-report.json" in noxfile


def test_starlight_release_readiness_doc_is_in_maintainer_navigation() -> None:
    """The Astro/Starlight site should mirror the release-readiness guidance."""
    doc = STARLIGHT_DOC.read_text(encoding="utf-8")
    latest_doc = STARLIGHT_LATEST_DOC.read_text(encoding="utf-8")
    config = STARLIGHT_CONFIG.read_text(encoding="utf-8")

    for content in (doc, latest_doc):
        assert "uv run nox -s release_readiness" in content
        assert "release candidate" in content.lower()
        assert "release-ready" in content.lower()
        assert "external acceptance" in content.lower()

    assert "Release Readiness" in config
    assert "/maintainers/release-readiness/" in config
    assert not Path("docs/source/release_readiness.rst").exists()
