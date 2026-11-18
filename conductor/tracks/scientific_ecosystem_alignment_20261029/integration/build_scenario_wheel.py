"""Build a development wheel and bind the real build to Arrow qualification.

This executor records observed inputs/output; it is not a signed attestation or
a release builder. Shared CI may adopt it together with the existing qualifier.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("scenario_arrow_qualifier", ROOT / "scripts/qualify_scenario_arrow.py")
qualifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qualifier)


def git(repository: Path, *arguments: str) -> str:
    """Read the selected checkout without modifying its history."""
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True)


def tool_version(name: str) -> str:
    """Record the compiler/build executable actually selected by PATH."""
    return subprocess.check_output([name, "--version"], text=True).strip()


def validate_build_tools(repository: Path, python_version: tuple[int, ...], maturin: str) -> None:
    """Require the declared Python and exact build backend before execution."""
    if tuple(python_version[:3]) != (3, 15, 2):
        raise ValueError("Python 3.15.2 is required")
    manifest = tomllib.loads((repository / "pyproject.toml").read_text())
    if manifest["build-system"]["requires"] != [f"maturin=={maturin}"]:
        raise ValueError("installed maturin must match the exact build-system requirement")


def kairos_revision(repository: Path) -> str:
    """Record a single immutable provider pin, never a branch or guessed SHA."""
    manifest = tomllib.loads((repository / "rust/crates/innovate-core/Cargo.toml").read_text())
    revisions = {
        value.get("rev")
        for value in manifest.get("dependencies", {}).values()
        if isinstance(value, dict) and value.get("git") == "https://github.com/edithatogo/kairos.git"
    }
    if len(revisions) != 1:
        raise ValueError("require one immutable Kairos revision")
    revision = revisions.pop()
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("require one immutable Kairos revision")
    return revision


def require_committed_inputs(repository: Path) -> None:
    """Reject modified, deleted and untracked native/build inputs."""
    paths = git(repository, "diff", "--name-only", "-z", "HEAD").split("\0")
    paths += git(repository, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    changed = sorted({p for p in paths if p in qualifier.BUILD_ROOTS or p.startswith(("rust/", "bindings/python/"))})
    if changed:
        raise ValueError(f"uncommitted build inputs: {changed}")


def run_build(command: list[str], repository: Path, log, environment: dict[str, str]) -> int:
    """Retain real combined output and the builder's exit status."""
    return subprocess.run(command, cwd=repository, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode


def build(repository: Path, output: Path, jobs: int) -> dict[str, object]:
    """Build once into a new directory and emit a record only after success."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("build jobs must be a positive integer")
    repository, output = repository.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require_committed_inputs(repository)
    validate_build_tools(repository, sys.version_info, importlib.metadata.version("maturin"))
    inputs = qualifier.build_inputs(repository)
    command = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "--locked",
        "--interpreter",
        sys.executable,
        "--out",
        str(output / "wheels"),
    ]
    environment = os.environ.copy()
    environment["CARGO_BUILD_JOBS"] = str(jobs)
    record = {
        "schema_version": 1,
        "source_commit": git(repository, "rev-parse", "HEAD").strip(),
        "source_tree": git(repository, "rev-parse", "HEAD^{tree}").strip(),
        "worktree_changes": git(repository, "status", "--short", "--untracked-files=all").splitlines(),
        "source_inputs": inputs,
        "kairos_revision": kairos_revision(repository),
        "rustc": tool_version("rustc"),
        "cargo": tool_version("cargo"),
        "python": sys.version,
        "maturin": importlib.metadata.version("maturin"),
        "build_profile": "dev",
        "build_jobs": jobs,
        "command_argv": command,
        "command": shlex.join(command),
        "executor_sha256": qualifier.sha256(Path(__file__)),
        "qualifier_sha256": qualifier.sha256(ROOT / "scripts/qualify_scenario_arrow.py"),
        "environment": {
            key: environment[key]
            for key in ("CARGO_HOME", "CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "RUSTUP_TOOLCHAIN", "RUSTC_WRAPPER")
            if key in environment
        },
        "claim": "Observed development-wheel build; not a release artifact or signed attestation.",
    }
    start = time.monotonic()
    log_path = output / "build.log"
    with log_path.open("w") as log:
        code = run_build(command, repository, log, environment)
    if code != 0:
        raise ValueError(f"build failed with exit {code}; see {log_path}")
    if qualifier.build_inputs(repository) != inputs:
        raise ValueError("native inputs changed during build")
    require_committed_inputs(repository)
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise ValueError("successful build must produce exactly one regular wheel")
    record.update(
        {
            "status": "built",
            "build_exit_code": code,
            "build_elapsed_seconds": round(time.monotonic() - start, 3),
            "build_log": "build.log",
            "build_log_sha256": qualifier.sha256(log_path),
            "source_inputs_unchanged_after_build": True,
            "wheel_filename": wheels[0].name,
            "wheel_sha256": qualifier.sha256(wheels[0]),
        }
    )
    qualifier.validate_build_record(record, inputs, record["wheel_sha256"])
    (output / "build-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    """Return failure without fabricating a successful provenance record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    try:
        result = build(arguments.repository, arguments.output, arguments.jobs)
    except (OSError, ValueError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "source_commit", "wheel_filename", "wheel_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
