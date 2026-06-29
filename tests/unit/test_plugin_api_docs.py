"""Tests for plugin and stability documentation."""

from __future__ import annotations

from pathlib import Path


def test_plugin_api_docs_pages_are_present() -> None:
    """The new contract docs should be present in the source tree."""
    docs_root = Path("docs/astro-site/src/content/docs/maintainers")
    tutorials_root = Path("docs/astro-site/src/content/docs/tutorials")
    starlight_config = Path("docs/astro-site/starlight.config.mjs").read_text()

    assert (docs_root / "plugins.md").is_file()
    assert (docs_root / "stability.md").is_file()
    assert (tutorials_root / "plugin-api-stability.md").is_file()
    assert not Path("docs/source/tutorials/plugin_api_stability.rst").exists()
    assert "/maintainers/plugins/" in starlight_config
    assert "/maintainers/stability/" in starlight_config
    assert "/tutorials/plugin-api-stability/" in starlight_config


def test_plugin_api_docs_describe_tiers_and_extension_points() -> None:
    """The tutorial should explain the public tiers and extension contract."""
    tutorial = Path("docs/astro-site/src/content/docs/tutorials/plugin-api-stability.md").read_text()

    assert "stable" in tutorial.lower()
    assert "provisional" in tutorial.lower()
    assert "internal" in tutorial.lower()
    assert "model_registry" in tutorial
    assert "serialization_adapter" in tutorial
    assert "release notes" in tutorial.lower()


def test_docs_indices_include_plugin_contract_pages() -> None:
    """The docs index should surface the new stability and plugin pages."""
    package_docs = Path("docs/source/innovate.rst").read_text()
    index_docs = Path("docs/astro-site/starlight.config.mjs").read_text()
    tutorials_docs = Path("docs/source/tutorials.rst").read_text()

    assert "maintainers/plugins.md" in package_docs
    assert "maintainers/stability.md" in package_docs
    assert "/maintainers/plugins/" in index_docs
    assert "/maintainers/stability/" in index_docs
    assert "/tutorials/plugin-api-stability/" in index_docs
    assert "tutorials/plugin_api_stability" not in tutorials_docs
