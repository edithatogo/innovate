"""Tests for Rust binding documentation coverage."""

from __future__ import annotations

from pathlib import Path


def test_rust_binding_readme_mentions_tracing_and_fallbacks() -> None:
    """The Rust binding README should mention runtime tracing behavior."""
    readme = Path("bindings/rust/README.md").read_text()

    assert "tracing" in readme
    assert "Fallback paths and bridge failures" in readme
    assert "native Rust" in readme
