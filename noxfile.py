"""Nox sessions for local and CI Python quality gates."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nox

SUPPORTED_PYTHONS = ("3.14",)
DEFAULT_PYTHON = "3.14"
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
nox.options.sessions = (
    "lint",
    "types",
    "tests",
    "coverage",
    "docs",
    "package",
    "security",
    "mutation",
    "version_sync",
    "dependency_dashboard",
    "binding_conformance",
    "release_supply_chain",
    "release_reproducibility",
    "release_readiness",
)


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
    """Run the required unit gate on the Python 3.14 baseline."""
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
    """Run strict basedpyright checks for the verified public API surface."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "basedpyright", "--warnings", *PUBLIC_API_TYPE_TARGETS, *session.posargs)


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Build the Astro/Starlight documentation smoke target."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    session.run("pnpm", "--dir", "docs/astro-site", "install", "--frozen-lockfile", external=True)
    session.run(
        "pnpm",
        "--dir",
        "docs/astro-site",
        "build",
        external=True,
        env={"STARLIGHT_POLYGLOT_PYTHON": "uv run python"},
    )


@nox.session(python=False)
def production_docs(session: nox.Session) -> None:
    """Verify the Astro/Starlight production documentation contract."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "python", "scripts/validate_examples.py", "--json")
    _run_uv(session, "run", "python", "scripts/generate_docs_dashboards.py", "--json")
    _run_uv(session, "run", "python", "scripts/verify_production_docs.py", "--json", *session.posargs)


@nox.session(python=False)
def examples(session: nox.Session) -> None:
    """Validate and classify documentation examples and snippets."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "python", "scripts/validate_examples.py", "--json", *session.posargs)


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
def security(session: nox.Session) -> None:
    """Run Bandit static analysis and Safety vulnerability scan on Python dependencies."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(session, "run", "bandit", "-r", "src/innovate", "-c", "pyproject.toml")
    _run_uv(session, "run", "safety", "check")


@nox.session(python=False)
def version_sync(session: nox.Session) -> None:
    """Check or rewrite release-version metadata across package manifests."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    script = "scripts/sync_versions.py"
    if "--write" in session.posargs:
        _run_uv(session, "run", "python", script, "--write")
        return
    _run_uv(session, "run", "python", script, "--check")


@nox.session(python=False)
def release_readiness(session: nox.Session) -> None:
    """Generate the local release-readiness report artifacts."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "scripts/release_readiness.py",
        "--json",
        "--allow-blocked",
        "--output",
        "docs/source/_static/release_readiness/readiness-report.json",
        *session.posargs,
    )


@nox.session(python=False)
def release_supply_chain(session: nox.Session) -> None:
    """Generate offline supply-chain release evidence."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "scripts/release_supply_chain.py",
        "--json",
        *session.posargs,
    )


@nox.session(python=False)
def release_reproducibility(session: nox.Session) -> None:
    """Generate deterministic reproducibility release evidence."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "scripts/release_reproducibility.py",
        "--json",
        *session.posargs,
    )


@nox.session(python=False)
def coverage(session: nox.Session) -> None:
    """Run unit tests, produce a standalone coverage report (HTML + XML), and write release evidence."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "-m",
        "pytest",
        "tests/unit/",
        "-m",
        "not optional_backend",
        *PYTEST_BASE_IGNORES,
        "--cov=innovate",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-json-report=.coverage-raw.json",
        "--cov-fail-under=80",
        "--tb=short",
        "-q",
        env={"CI": "true", "JAX_PLATFORM_NAME": "cpu"},
    )
    # Write coverage evidence for release readiness and enforce threshold
    _run_uv(
        session,
        "run",
        "python",
        "-c",
        (
            "import json, sys; "
            "from scripts.release_evidence import write_coverage_evidence, COVERAGE_THRESHOLD_LINE_RATE; "
            "data = json.loads(open('.coverage-raw.json').read()); "
            "meta = data.get('meta', {}); "
            "totals = data.get('totals', {}); "
            "line_rate = totals.get('percent_covered', 0.0) / 100.0; "
            "write_coverage_evidence("
            "line_rate=line_rate, "
            "branch_rate=totals.get('percent_covered_branches', 0.0) / 100.0, "
            "lines_covered=totals.get('covered_lines', 0), "
            "lines_valid=totals.get('num_statements', 0), "
            "branches_covered=totals.get('covered_branches', 0), "
            "branches_valid=totals.get('num_branches', 0), "
            "summary=f\"Line rate: {totals.get('percent_covered', 0):.1f}%\", "
            "); "
            "print('Coverage evidence written'); "
            "if line_rate < COVERAGE_THRESHOLD_LINE_RATE: "
            "    print(f'FAIL: Line rate {line_rate:.1%} is below threshold {COVERAGE_THRESHOLD_LINE_RATE:.0%}', file=sys.stderr); "
            "    sys.exit(1); "
            "print(f'Line rate {line_rate:.1%} meets threshold {COVERAGE_THRESHOLD_LINE_RATE:.0%}')"
        ),
    )


@nox.session(python=False)
def mutation(session: nox.Session) -> None:
    """Run mutmut mutation testing, enforce >70% threshold, and write release evidence."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "-m",
        "mutmut",
        "run",
        *session.posargs,
    )
    _run_uv(
        session,
        "run",
        "python",
        "-m",
        "mutmut",
        "results",
    )
    # Parse mutmut results and write evidence
    _run_uv(
        session,
        "run",
        "python",
        "-c",
        (
            "from scripts.release_evidence import _parse_mutmut_results, write_mutation_evidence, MUTATION_SCORE_THRESHOLD; "
            "parsed = _parse_mutmut_results(); "
            "score = parsed['score']; "
            "write_mutation_evidence("
            "score=score, "
            "mutants_killed=parsed['killed'], "
            "mutants_total=parsed['total'], "
            "summary=f\"Mutants killed: {parsed['killed']}/{parsed['total']} (score: {score:.1%})\", "
            "); "
            "if score < MUTATION_SCORE_THRESHOLD: "
            "    import sys; "
            "    print(f'FAIL: Mutation score {score:.1%} is below threshold {MUTATION_SCORE_THRESHOLD:.0%}', file=sys.stderr); "
            "    sys.exit(1); "
            "print(f'Mutation score {score:.1%} meets threshold {MUTATION_SCORE_THRESHOLD:.0%}')"
        ),
    )


@nox.session(python=False)
def dependency_dashboard(session: nox.Session) -> None:
    """Generate a non-mutating dependency freshness dashboard across all ecosystems."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "scripts/dependency_dashboard.py",
        "--json",
        *session.posargs,
    )


@nox.session(python=False)
def binding_conformance(session: nox.Session) -> None:
    """Run polyglot binding conformance tests (Python, Rust, R, Julia, C#, TS)."""
    _run_uv(session, "sync", "--python", DEFAULT_PYTHON)
    _run_uv(
        session,
        "run",
        "python",
        "-m",
        "pytest",
        "tests/unit/test_polyglot_binding_conformance.py",
        "tests/unit/test_polyglot_binding_golden_fixtures.py",
        "tests/unit/test_polyglot_binding_hardening.py",
        "tests/unit/test_binding_conformance_ci.py",
        "-q",
        "--tb=short",
        *session.posargs,
    )
