"""Tests for Conductor workflow and tech-stack governance decisions."""

from __future__ import annotations

from pathlib import Path


def test_python_task_runner_remains_uv_first() -> None:
    """The repo should keep uv as the primary Python task runner."""
    tech_stack = Path("conductor/tech-stack.md").read_text()
    workflow = Path("conductor/workflow.md").read_text()

    assert "Python task orchestration remains `uv`-first" in tech_stack
    assert "nox` is intentionally" in tech_stack
    assert "`uv` is the canonical Python runner" in workflow
    assert "A `nox` layer is not" in workflow
