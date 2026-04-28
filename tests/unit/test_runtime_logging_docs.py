"""Tests for the runtime logging and instrumentation guidance."""

from __future__ import annotations

from pathlib import Path


def test_runtime_logging_docs_are_present() -> None:
    """The observability guidance should be documented as a first-class page."""
    docs_root = Path("docs/source")

    assert (docs_root / "runtime_logging.rst").is_file()


def test_runtime_logging_docs_define_the_repo_policy() -> None:
    """The docs should distinguish runtime logging from test/example printing."""
    docs = Path("docs/source/runtime_logging.rst").read_text()

    assert "standard logging primitives" in docs
    assert "structured error payloads" in docs
    assert "Library modules should create module-level loggers" in docs
    assert "Bridge scripts should keep stdout machine-readable" in docs
    assert "Tests, examples, and intentionally human-facing scripts may still use" in docs
    assert "Rust-native runtime observability" in docs
