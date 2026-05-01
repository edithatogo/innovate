"""Tests for DataFrame engine experiment documentation."""

from __future__ import annotations

from pathlib import Path

import tomllib


def test_dataframe_engine_docs_are_published() -> None:
    """The DataFrame engine strategy should be visible in the Sphinx docs."""
    docs_root = Path("docs/source")
    index = (docs_root / "index.rst").read_text()
    package_docs = (docs_root / "innovate.rst").read_text()

    assert (docs_root / "dataframe_engine_experiments.rst").is_file()
    assert "dataframe_engine_experiments" in index
    assert "docs/source/dataframe_engine_experiments.rst" in package_docs


def test_dataframe_engine_docs_define_support_and_promotion_boundaries() -> None:
    """Docs should distinguish optional engines from the stable tabular contract."""
    docs = Path("docs/source/dataframe_engine_experiments.rst").read_text()

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
