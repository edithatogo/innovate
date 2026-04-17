"""Tests for plugin and stability documentation."""

from __future__ import annotations

from pathlib import Path


def test_plugin_api_docs_pages_are_present() -> None:
    """The new contract docs should be present in the source tree."""
    docs_root = Path("docs/source")

    assert (docs_root / "innovate.plugins.rst").is_file()
    assert (docs_root / "innovate.stability.rst").is_file()
    assert (docs_root / "tutorials/plugin_api_stability.rst").is_file()


def test_plugin_api_docs_describe_tiers_and_extension_points() -> None:
    """The tutorial should explain the public tiers and extension contract."""
    tutorial = Path("docs/source/tutorials/plugin_api_stability.rst").read_text()

    assert "stable" in tutorial.lower()
    assert "provisional" in tutorial.lower()
    assert "internal" in tutorial.lower()
    assert "model_registry" in tutorial
    assert "serialization_adapter" in tutorial
    assert "release notes" in tutorial.lower()


def test_docs_indices_include_plugin_contract_pages() -> None:
    """The docs index should surface the new stability and plugin pages."""
    package_docs = Path("docs/source/innovate.rst").read_text()
    index_docs = Path("docs/source/index.rst").read_text()
    tutorials_docs = Path("docs/source/tutorials.rst").read_text()

    assert "innovate.plugins" in package_docs
    assert "innovate.stability" in package_docs
    assert "innovate.plugins.rst" in index_docs
    assert "innovate.stability.rst" in index_docs
    assert "tutorials/plugin_api_stability" in tutorials_docs
