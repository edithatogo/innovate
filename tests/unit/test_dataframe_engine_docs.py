"""Tests for DataFrame engine experiment documentation."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dataframe_engine_docs_are_published() -> None:
    """The DataFrame engine strategy should be visible in the Starlight docs."""
    docs_root = Path("docs/astro-site/src/content/docs/roadmap")
    index = Path("docs/astro-site/starlight.config.mjs").read_text()
    package_docs = Path("docs/source/innovate.rst").read_text()

    assert (docs_root / "dataframe-engine.md").is_file()
    assert "/roadmap/dataframe-engine/" in index
    assert "roadmap/dataframe-engine.md" in package_docs


def test_dataframe_engine_docs_define_support_and_promotion_boundaries() -> None:
    """Docs should distinguish optional engines from the stable tabular contract."""
    docs = Path("docs/astro-site/src/content/docs/roadmap/dataframe-engine.md").read_text()

    for phrase in (
        "pandas plus PyArrow remains the default",
        "Polars is experimental",
        "Kernel schemas and Arrow-compatible payloads remain the public contract",
        "Polars lazy query plans are not a public contract",
        "XLA-backed numerical kernels",
        "benchmark evidence",
        "fallback",
    ):
        assert phrase in docs


def test_polars_is_declared_as_an_optional_dataframe_dependency() -> None:
    """Polars should be installable for experiments without becoming required."""
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]

    assert not any(dependency.startswith("polars") for dependency in dependencies)
    assert any(dependency.startswith("polars") for dependency in extras["dataframe"])
