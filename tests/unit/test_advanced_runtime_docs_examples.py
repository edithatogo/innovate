"""Tests for advanced runtime documentation and examples."""

from __future__ import annotations

import runpy
from pathlib import Path

EXAMPLE_PATH = Path("examples/advanced_runtime_workflows.py")
SPHINX_DOC = Path("docs/source/tutorials/advanced_runtime_workflows.rst")
STARLIGHT_DOC = Path("docs/astro-site/src/content/docs/tutorials/advanced-runtime.md")
STARLIGHT_LATEST_DOC = Path("docs/astro-site/src/content/docs/latest/tutorials/advanced-runtime.md")
TUTORIAL_INDEX = Path("docs/source/tutorials.rst")


def test_advanced_runtime_example_executes_and_returns_payloads() -> None:
    """The end-to-end example should run in CI without optional dependencies."""
    namespace = runpy.run_path(str(EXAMPLE_PATH))

    report = namespace["build_report"]()
    assert set(report) == {
        "ensemble",
        "policy",
        "streaming",
        "calibration",
    }
    assert report["ensemble"]["capability"]["workflow"] == "regime_ensemble"
    assert report["policy"]["metadata"]["incremental_effect"] > 0
    assert report["streaming"]["metadata"]["state"]["last_observed"] > 0
    assert report["calibration"]["diagnostics"]["coverage"] >= 0.8


def test_advanced_runtime_sphinx_docs_are_indexed() -> None:
    """Legacy Sphinx docs should retain an indexed tutorial for the advanced runtime."""
    doc = SPHINX_DOC.read_text(encoding="utf-8")
    index = TUTORIAL_INDEX.read_text(encoding="utf-8")

    assert "Advanced runtime workflows" in doc
    assert "examples/advanced_runtime_workflows.py" in doc
    assert "performance_evidence.json" in doc
    assert "tutorials/advanced_runtime_workflows" in index


def test_advanced_runtime_starlight_docs_exist_for_current_and_latest() -> None:
    """Starlight current and latest routes should document the advanced runtime."""
    for path in [STARLIGHT_DOC, STARLIGHT_LATEST_DOC]:
        text = path.read_text(encoding="utf-8")
        assert "Advanced runtime workflows" in text
        assert "Stable surfaces" in text
        assert "Experimental surfaces" in text
        assert "performance_evidence.json" in text
