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
