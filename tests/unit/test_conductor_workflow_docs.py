"""Tests for Conductor workflow and tech-stack governance decisions."""

from __future__ import annotations

from pathlib import Path


def test_python_task_runner_remains_uv_first() -> None:
    """The repo should keep uv primary while exposing nox orchestration."""
    tech_stack = Path("conductor/tech-stack.md").read_text()
    workflow = Path("conductor/workflow.md").read_text()

    assert "Python dependency management remains `uv`-first" in tech_stack
    assert "**nox** — Python task orchestration" in tech_stack
    assert "`uv` is the canonical Python dependency manager" in workflow
    assert "`nox` provides the Python task layer" in workflow


def test_rust_profiling_stack_records_cpu_memory_and_gpu_scope() -> None:
    """The tech stack should record Rust profiling tools and GPU ownership."""
    tech_stack = Path("conductor/tech-stack.md").read_text()

    assert "**cargo-flamegraph**" in tech_stack
    assert "**DHAT**" in tech_stack
    assert "JAX/XLA device profilers" in tech_stack
    assert "until Rust owns a promoted native GPU execution backend" in tech_stack
