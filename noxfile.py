"""Native development gates; full scientific migration remains separately blocked."""

import os
from pathlib import Path

import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["lint", "types", "tests", "native", "package"]
ENVIRONMENT = Path(
    os.environ.get("UV_PROJECT_ENVIRONMENT", ".venv-native" if Path(".venv-native").exists() else ".venv")
).resolve()
BIN = ENVIRONMENT / ("Scripts" if os.name == "nt" else "bin")
PYTHON = str(BIN / ("python.exe" if os.name == "nt" else "python"))


def tool(session, name, *args):
    session.run(str(BIN / name), *args, external=True)


@nox.session(python=False)
def lint(session):
    tool(session, "ruff", "check", "bindings/python", "tests/native", "noxfile.py", "scripts/check_migration.py")
    tool(
        session,
        "ruff",
        "format",
        "--check",
        "bindings/python",
        "tests/native",
        "noxfile.py",
        "scripts/check_migration.py",
    )
    session.run("cargo", "fmt", "--all", "--check", external=True)


@nox.session(python=False)
def types(session):
    tool(session, "basedpyright", "--pythonpath", PYTHON)


@nox.session(python=False)
def tests(session):
    session.run(PYTHON, "-m", "pytest", "tests/native", *session.posargs, external=True)


@nox.session(python=False)
def native(session):
    session.run(
        "cargo",
        "test",
        "--locked",
        "-p",
        "innovate-core",
        "-p",
        "innovate-tabular",
        "-p",
        "innovate-ffi",
        "-j",
        "2",
        external=True,
    )


@nox.session(python=False)
def package(session):
    tool(session, "maturin", "build", "--locked", "--interpreter", PYTHON)


@nox.session(python=False)
def release_readiness(session):
    session.run(PYTHON, "scripts/check_migration.py", "--release", external=True)


@nox.session(python=False)
def docs(session):
    session.run(
        "npx",
        "--yes",
        "pnpm@12.18.0",
        "--dir",
        "docs/astro-site",
        "build",
        external=True,
        env={"STARLIGHT_POLYGLOT_PYTHON": PYTHON},
    )


@nox.session(python=False)
def binding_conformance(session):
    native(session)
    tests(session)


@nox.session(python=False)
def release_supply_chain(session):
    release_readiness(session)


@nox.session(python=False)
def release_reproducibility(session):
    release_readiness(session)


@nox.session(python=False)
def security(session):
    tool(session, "bandit", "-r", "bindings/python/innovate", "-q")
    session.run("cargo", "audit", "--file", "Cargo.lock", external=True)


@nox.session(python=False)
def version_sync(session):
    session.run(PYTHON, "scripts/sync_versions.py", "--check", external=True)


@nox.session(python=False)
def dependency_dashboard(session):
    session.run(PYTHON, "scripts/dependency_dashboard.py", *session.posargs, external=True)
