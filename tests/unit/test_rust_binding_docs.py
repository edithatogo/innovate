"""Tests for Rust binding documentation coverage."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


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


def test_rust_crate_includes_and_documents_profiling_files() -> None:
    """The Rust crate package should ship documented profiling entry points."""
    cargo = tomllib.loads(Path("bindings/rust/Cargo.toml").read_text())
    readme = Path("bindings/rust/README.md").read_text()

    include = set(cargo["package"]["include"])

    assert "benches/**" in include
    assert "examples/**" in include
    assert "scripts/**" in include

    for package_file in (
        "benches/native_kernel.rs",
        "examples/profile_memory_native_kernels.rs",
        "scripts/profile_native_kernels.sh",
        "scripts/profile_memory_native_kernels.sh",
    ):
        assert package_file in readme

    assert "crate package intentionally includes the profiling surface" in readme
