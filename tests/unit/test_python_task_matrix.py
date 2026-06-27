"""Governance checks for Python version and task orchestration policy."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

SUPPORTED_PYTHONS = ("3.14",)
LOCK_VALIDATED_RUNTIME_DEPENDENCIES = {
    "jitcdde": "jitcdde>=1.8.3,<2",
    "mesa": "mesa>=3.5.1,<4",
    "ndlib": "ndlib>=5.1.1,<6",
    "networkx": "networkx>=3.6.1,<4",
    "numpy": "numpy>=2.4.4,<3",
    "pandas": "pandas>=3.0.2,<4",
    "pyarrow": "pyarrow>=23.0.1,<24",
    "pymannkendall": "pymannkendall>=1.4.3,<2",
    "pytensor": "pytensor>=2.38.2,<3",
    "ruptures": "ruptures>=1.1.9,<1.1.10",
    "scipy": "scipy>=1.17.1,<2",
    "statsmodels": "statsmodels>=0.14.6,<0.15",
    "sympy": "sympy>=1.14,<2",
}


def test_ci_runs_required_unit_gate_on_supported_python_versions() -> None:
    """The required unit gate should cover the Python 3.14 baseline."""
    ci = Path(".github/workflows/ci.yml").read_text()

    matrix_match = re.search(r"python-version:\s*\[(?P<versions>[^\]]+)\]", ci)

    assert matrix_match is not None
    assert tuple(re.findall(r'"(\d+\.\d+)"', matrix_match.group("versions"))) == SUPPORTED_PYTHONS
    assert "uv sync --python ${{ matrix.python-version }}" in ci
    assert "uv run --python ${{ matrix.python-version }} python -m pytest" in ci
    assert "uv run nox --list" in ci


def test_pyproject_declares_nox_and_python_314_format_policy() -> None:
    """The Python toolchain should include nox and the strict type gate."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dev_dependencies = pyproject["dependency-groups"]["dev"]
    classifiers = pyproject["project"]["classifiers"]

    assert any(dependency.startswith("nox>=") for dependency in dev_dependencies)
    assert any(dependency.startswith("basedpyright>=") for dependency in dev_dependencies)
    assert any(dependency.startswith("pydantic>=2.") and "<3" in dependency for dependency in dev_dependencies)
    assert not any(dependency.startswith(("ty>=", "mypy>=")) for dependency in dev_dependencies)
    assert not any(dependency.startswith("pydantic<2") for dependency in dev_dependencies)
    assert pyproject["project"]["requires-python"] == ">=3.14"
    assert pyproject["tool"]["pyproject-fmt"]["max_supported_python"] == "3.14"
    assert {f"Programming Language :: Python :: {version}" for version in SUPPORTED_PYTHONS}.issubset(classifiers)
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert pyproject["tool"]["basedpyright"]["typeCheckingMode"] == "strict"
    assert pyproject["tool"]["basedpyright"]["pythonVersion"] == "3.14"
    assert pyproject["tool"]["basedpyright"]["reportUnsupportedDunderAll"] is False


def test_pyproject_runtime_floors_match_python_314_lock_policy() -> None:
    """Runtime dependency floors should mirror the Python 3.14 lock-validated stack."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = {
        re.match(r"^[A-Za-z0-9_.-]+", dependency).group(0): dependency
        for dependency in pyproject["project"]["dependencies"]
    }
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert dependencies == LOCK_VALIDATED_RUNTIME_DEPENDENCIES
    assert "numpyro>=0.20.1,<0.21" in optional_dependencies["jax"]
    assert "numpyro>=0.20.1,<0.21" in optional_dependencies["bayesian"]


def test_pyproject_is_the_only_root_python_dependency_manifest() -> None:
    """Avoid maintaining a root requirements.txt that can drift from pyproject.toml."""
    assert not Path("requirements.txt").exists()


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
    assert assignments["DEFAULT_PYTHON"] == "3.14"
    assert {"tests", "optional_backends", "lint", "types", "docs", "package", "version_sync"}.issubset(sessions)
    assert 'basedpyright", "--warnings"' in Path("noxfile.py").read_text()


def test_noxfile_includes_dependency_dashboard_session() -> None:
    """Nox should have a dependency_dashboard session that runs the dependency dashboard script."""
    module = ast.parse(Path("noxfile.py").read_text())
    session_names = {
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
    assert "dependency_dashboard" in session_names
    nox_text = Path("noxfile.py").read_text()
    assert "scripts/dependency_dashboard.py" in nox_text


def test_noxfile_includes_binding_conformance_session() -> None:
    """Nox should have a bindings or binding_conformance session for polyglot checks."""
    module = ast.parse(Path("noxfile.py").read_text())
    session_names = {
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
    assert "binding_conformance" in session_names


def test_noxfile_includes_mutation_session() -> None:
    """Nox should have a mutation session that runs mutmut."""
    module = ast.parse(Path("noxfile.py").read_text())
    session_names = {
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
    assert "mutation" in session_names
    nox_text = Path("noxfile.py").read_text()
    assert "mutmut" in nox_text


def test_noxfile_includes_coverage_session() -> None:
    """Nox should have a coverage session that produces a standalone coverage report."""
    module = ast.parse(Path("noxfile.py").read_text())
    session_names = {
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
    assert "coverage" in session_names
    nox_text = Path("noxfile.py").read_text()
    assert "--cov-report=html" in nox_text or "coverage html" in nox_text


def test_nox_default_sessions_include_all_required_gates() -> None:
    """nox.options.sessions should list every required quality gate."""
    module = ast.parse(Path("noxfile.py").read_text())
    options_sessions = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "sessions":
                    if isinstance(target.value, ast.Attribute) and target.value.attr == "options":
                        if isinstance(target.value.value, ast.Name) and target.value.value.id == "nox":
                            options_sessions = {ast.literal_eval(e) for e in node.value.elts}
    assert options_sessions is not None
    required = {"lint", "types", "tests", "docs", "package", "security",
                "version_sync", "coverage", "mutation",
                "dependency_dashboard", "binding_conformance"}
    assert required.issubset(options_sessions)
