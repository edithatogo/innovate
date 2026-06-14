"""Nox sessions for local and CI Python quality gates."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nox

SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
DEFAULT_PYTHON = "3.12"
PYTEST_BASE_IGNORES = (
    "--ignore=tests/unit/test_bayesian_fitter_robust.py",
    "--ignore=tests/unit/test_blackjax_fitter.py",
    "--ignore=tests/unit/test_dynamics_competition_direct.py",
    "--ignore=tests/unit/test_dynamics_contagion_direct.py",
    "--ignore=tests/unit/test_categorization.py",
    "--ignore=tests/unit/test_path_dependence.py",
    "--ignore=tests/unit/test_preprocess.py",
    "--ignore=tests/unit/test_plots_diagnostics.py",
    "--ignore=tests/unit/test_plots_network.py",
    "--ignore=tests/unit/test_advanced_functionality_comprehensive.py",
    "--ignore=tests/unit/test_curve_fitter.py",
)
PUBLIC_API_TYPE_TARGETS = (
    "src/innovate/__init__.py",
    "src/innovate/backend.py",
    "src/innovate/backends/__init__.py",
    "src/innovate/capabilities.py",
    "src/innovate/diffuse/__init__.py",
    "src/innovate/substitute/__init__.py",
    "src/innovate/ecosystem/__init__.py",
)

nox.options.default_venv_backend = "none"
nox.options.sessions = ("lint", "types", "tests", "docs", "package", "version_sync")


def _run_uv(session: nox.Session, *args: str, env: dict[str, str] | None = None) -> None:
    session.run("uv", *args, external=True, env=env)


def _prepare_python(session: nox.Session, python: str, *sync_args: str) -> None:
    _run_uv(session, "python", "install", python)
    _run_uv(session, "sync", "--python", python, *sync_args)
    _run_uv(session, "pip", "install", "-e", ".")


def _pytest_args(test_targets: Sequence[str]) -> tuple[str, ...]:
    return (
        "run",
        "python",
        "-m",
        "pytest",
        *(test_targets or ("tests/unit/",)),
        "-m",
        "not optional_backend",
        *PYTEST_BASE_IGNORES,
        "--cov=innovate",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--cov-fail-under=0",
        "--tb=short",
        "-q",
    )


@nox.session(python=False)
@nox.parametrize("python", SUPPORTED_PYTHONS)
def tests(session: nox.Session, python: str) -> None:
    """Run the required unit gate on every supported Python version."""
    _prepare_python(session, python)
    _run_uv(
        session,
        *_pytest_args(session.posargs),
        env={"CI": "true", "JAX_PLATFORM_NAME": "cpu"},
    )


@nox.session(python=False)
def optional_backends(session: nox.Session) -> None:
    """Run optional JAX/Bayesian backend tests on the default CI Python."""
    _prepare_python(session, DEFAULT_PYTHON, "--extra", "jax", "--extra", "bayesian")
    _run_uv(
        session,
        "run",
        "python",
        "-m",
        "pytest",
        *(session.posargs or ("tests/unit/",)),
        "-m",
        "optional_backend",
        "--cov=innovate",
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--cov-fail-under=0",
        "--tb=short",
        "-q",
        env={"CI": "true", "JAX_PLATFORM_NAME": "cpu"},
    )


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Run Ruff linting and formatting checks."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "ruff", "check", ".", *session.posargs)
    _run_uv(session, "run", "ruff", "format", "--check", ".")


@nox.session(python=False)
def types(session: nox.Session) -> None:
    """Run type checks for the verified public API surface."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "ty", "check", *PUBLIC_API_TYPE_TARGETS, *session.posargs)
    _run_uv(session, "run", "mypy", *PUBLIC_API_TYPE_TARGETS)


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Build the Sphinx documentation smoke target."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "python", "-m", "sphinx", "-b", "html", "docs/source", "/tmp/innovate-docs-build")


@nox.session(python=False)
def package(session: nox.Session) -> None:
    """Build the Python package artifacts."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "build", *session.posargs)
    _run_uv(session, "run", "twine", "check", "dist/*")
    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        session.error("No wheels found under dist/")
    _run_uv(session, "run", "check-wheel-contents", *(str(wheel) for wheel in wheels))


@nox.session(python=False)
def version_sync(session: nox.Session) -> None:
    """Check or rewrite release-version metadata across package manifests."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    script = "scripts/sync_versions.py"
    if "--write" in session.posargs:
        _run_uv(session, "run", "python", script, "--write")
        return
    _run_uv(session, "run", "python", script, "--check")
