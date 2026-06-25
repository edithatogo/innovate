"""Tests for the runtime logging and instrumentation guidance."""

from __future__ import annotations

from pathlib import Path


def test_runtime_logging_docs_are_present() -> None:
    """The observability guidance should be documented as a first-class page."""
    docs_root = Path("docs/astro-site/src/content/docs/maintainers")
    starlight_config = Path("docs/astro-site/starlight.config.mjs").read_text()

    assert (docs_root / "runtime-logging.md").is_file()
    assert "/maintainers/runtime-logging/" in starlight_config


def test_runtime_logging_docs_define_the_repo_policy() -> None:
    """The docs should distinguish runtime logging from test/example printing."""
    docs = Path("docs/astro-site/src/content/docs/maintainers/runtime-logging.md").read_text()

    assert "standard logging primitives" in docs
    assert "structured error payloads" in docs
    assert "Library modules should create module-level loggers" in docs
    assert "Bridge scripts should keep stdout machine-readable" in docs
    assert "Tests, examples, and intentionally human-facing scripts may still use" in docs
    assert "Rust-native runtime observability" in docs
