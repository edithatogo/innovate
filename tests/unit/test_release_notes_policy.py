"""Release notes and package-artifact documentation policy guards."""

from __future__ import annotations

from pathlib import Path


def test_changelog_and_release_policy_cover_current_version() -> None:
    """Release notes policy should cover the aligned package version."""
    pyproject = Path("pyproject.toml").read_text()
    changelog = Path("CHANGELOG.md").read_text()
    policy = Path("docs/astro-site/src/content/docs/maintainers/release-notes.md").read_text()

    assert 'version = "0.5.0"' in pyproject
    assert "## [0.5.0]" in changelog
    assert "## [0.4.0]" in changelog
    assert "Release Please" in policy
    assert "Release Drafter" in policy
    assert "Commitizen" in policy
    assert "CHANGELOG.md" in policy


def test_release_drafter_policy_comment_is_defined() -> None:
    """Release Drafter should not claim the release-note policy is undefined."""
    config = Path(".github/release-drafter.yml").read_text()

    assert "until the release notes policy is defined" not in config
    assert "Astro/Starlight maintainer docs" in config


def test_r_publication_docs_match_vignette_artifacts() -> None:
    """Binding publication docs should not drift from the R package source."""
    publication_docs = Path("docs/astro-site/src/content/docs/maintainers/publication.md").read_text()
    description = Path("bindings/r/DESCRIPTION").read_text()

    assert Path("bindings/r/vignettes/innovate-r-kernel.Rmd").is_file()
    assert "VignetteBuilder: knitr" in description
    assert "currently has no" not in publication_docs
    assert "source vignette" in publication_docs
    assert (
        "r-manual-${{ steps.r_metadata.outputs.package }}-${{ steps.r_metadata.outputs.version }}" in publication_docs
    )


def test_release_policy_is_in_docs_toctree() -> None:
    """The release notes policy should be reachable from Starlight navigation."""
    index = Path("docs/astro-site/starlight.config.mjs").read_text()

    assert "/maintainers/release-notes/" in index


def test_release_notes_policy_mentions_version_sync_guard() -> None:
    """Release policy should explain the canonical version sync path."""
    policy = Path("docs/astro-site/src/content/docs/maintainers/release-notes.md").read_text()

    assert "Version Synchronization" in policy
    assert "scripts/sync_versions.py" in policy
    assert "--check" in policy
    assert "--write" in policy
