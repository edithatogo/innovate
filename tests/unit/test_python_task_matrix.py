"""Governance checks for Python version and task orchestration policy."""

from __future__ import annotations

import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")


def test_ci_runs_required_unit_gate_on_supported_python_versions() -> None:
    """The required unit gate should cover every supported Python version."""
    ci = Path(".github/workflows/ci.yml").read_text()

    matrix_match = re.search(r"python-version:\s*\[(?P<versions>[^\]]+)\]", ci)

    assert matrix_match is not None
    assert tuple(re.findall(r'"(\d+\.\d+)"', matrix_match.group("versions"))) == SUPPORTED_PYTHONS
    assert "uv sync --python ${{ matrix.python-version }}" in ci
    assert "uv run --python ${{ matrix.python-version }} python -m pytest" in ci
    assert "uv run nox --list" in ci


def test_pyproject_declares_nox_and_python_314_format_policy() -> None:
    """The Python toolchain should include nox and know about Python 3.14."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dev_dependencies = pyproject["dependency-groups"]["dev"]
    classifiers = pyproject["project"]["classifiers"]

    assert any(dependency.startswith("nox>=") for dependency in dev_dependencies)
    assert pyproject["tool"]["pyproject-fmt"]["max_supported_python"] == "3.14"
    assert {f"Programming Language :: Python :: {version}" for version in SUPPORTED_PYTHONS}.issubset(classifiers)


def test_noxfile_exposes_expected_sessions_and_version_matrix() -> None:
    """Nox should be the local entrypoint for the Python quality gates."""
    module = ast.parse(Path("noxfile.py").read_text())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SUPPORTED_PYTHONS", "DEFAULT_PYTHON"}
    }
    sessions = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "session"
            for decorator in node.decorator_list
        )
    }

    assert assignments["SUPPORTED_PYTHONS"] == SUPPORTED_PYTHONS
    assert assignments["DEFAULT_PYTHON"] == "3.12"
    assert {"tests", "optional_backends", "lint", "types", "docs", "package"}.issubset(sessions)
