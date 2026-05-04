"""Tests for Rust binding documentation coverage."""

from __future__ import annotations

from pathlib import Path


def test_rust_binding_readme_mentions_tracing_and_fallbacks() -> None:
    """The Rust binding README should mention runtime tracing behavior."""
    readme = Path("bindings/rust/README.md").read_text()

    assert "tracing" in readme
    assert "Fallback paths and bridge failures" in readme
    assert "native Rust" in readme


def test_rust_binding_readme_mentions_memory_and_gpu_profiling_scope() -> None:
    """The Rust binding README should document memory profiling and GPU scope."""
    readme = Path("bindings/rust/README.md").read_text()

    assert "profile_memory_native_kernels.sh" in readme
    assert "DHAT" in readme
    assert "GPU profiling" in readme
    assert "not yet a Rust crate responsibility" in readme
    assert "not yet entirely Rust" in readme
