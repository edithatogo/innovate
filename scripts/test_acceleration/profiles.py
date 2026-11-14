"""Bounded local test profiles and conservative cache validation.

This runner never supplies CI or release acceptance. Python dependency tracing
cannot observe native execution, data-file reads or subprocess dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPES = {
    "native": ("tests/native",),
    "policy": ("tests/native/test_migration_policy.py", "tests/native/test_migration_mutation.py"),
}
VERSIONS = {"pytest": "9.2.0", "coverage": "7.17.2", "pytest-testmon": "2.3.0", "pytest-gremlins": "1.13.2"}


class ProfileError(ValueError):
    """The requested local profile cannot produce trustworthy feedback."""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root: Path, scope: str, environment: dict, *, native_paths=()) -> dict[str, str]:
    """Fingerprint every opaque input; leave binding Python edits to testmon.

    Test contents join the inventory digest so parametrization and new test cases
    also rebuild the database. All other files (including scripts, native sources,
    fixtures and evidence) invalidate it conservatively. Ignored outputs stay local.
    """
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        .decode()
        .split("\0")
    )
    inventory, opaque, python = [], [], []
    for name in sorted(set(names) - {""}):
        path = root / name
        if path.is_symlink():
            raise ProfileError(f"Symlinked source inputs are unsupported: {name}")
        exists = path.is_file()
        inventory.append((name, exists))
        if not exists:
            continue
        digest = _file_digest(path)
        if name.startswith("tests/") and path.suffix == ".py":
            inventory.append((name, digest))
        elif name.startswith("bindings/python/") and path.suffix == ".py":
            python.append((name, digest))
        else:
            opaque.append((name, digest))
    return {
        "environment": _digest({"scope": scope, "runtime": environment, "schema": 1}),
        "inventory": _digest(inventory),
        "opaque": _digest(opaque),
        "python": _digest(python),
        "native": _digest([_file_digest(path) for path in sorted(native_paths)]),
    }


def invalidation_reasons(previous: dict | None, current: dict) -> list[str]:
    """Return the inputs that require a complete dependency rebuild."""
    if not previous:
        return ["unseeded"]
    return [key for key in ("environment", "inventory", "opaque", "native") if previous.get(key) != current[key]]


def validate_environment(environ: dict, version: tuple) -> None:
    """Reject CI use and overlapping selectors, instrumentation and workers."""
    if version[:3] != (3, 15, 2):
        raise ProfileError("Use the selected repo-local Python 3.15.2 interpreter")
    for name in ("CI", "GITHUB_ACTIONS"):
        if environ.get(name, "").lower() not in ("", "0", "false"):
            raise ProfileError("Selective profiles are local feedback only; run the ordinary complete CI suite")
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_XDIST_WORKER",
        "COVERAGE_PROCESS_START",
        "COVERAGE_RCFILE",
        "COVERAGE_CORE",
        "COVERAGE_FILE",
        "COVERAGE_DEBUG",
        "COVERAGE_DEBUG_FILE",
        "TESTMON_DATAFILE",
        "TESTMON_ENVIRONMENT",
        "INNOVATE_TEST_LIVE_OECD",
    ):
        if environ.get(name):
            raise ProfileError(f"Unset incompatible inherited option {name}")


@contextmanager
def cache_lease(directory: Path):
    """Permit one writer; a stale lease requires deliberate local recovery."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "active.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ProfileError(f"Profile cache already in use: {lock}; never remove an active writer's lease") from exc
    try:
        yield
    finally:
        lock.rmdir()


def classify_result(mode: str, code: int, metrics: dict | None, *, cached_items=0, rebuilt=False) -> str:
    """Distinguish actual successful execution from a seeded no-selection run."""
    fields = ("collected", "selected", "executed", "errors", "failed", "skipped", "exitstatus", "passed")
    if (
        not metrics
        or any(type(metrics.get(key)) is not int or metrics[key] < 0 for key in fields)
        or metrics["errors"]
        or metrics["failed"]
        or code not in (0, 5)
    ):
        return "failed"
    if (
        mode == "testmon"
        and not rebuilt
        and cached_items > 0
        and metrics["selected"] == metrics["executed"] == metrics["skipped"] == 0
        and metrics["exitstatus"] in (0, 5)
    ):
        return "no_tests_selected"
    if code == metrics["exitstatus"] == 0 and metrics["passed"] > 0:
        return "passed"
    return "failed"


def runtime_inventory() -> dict:
    """Record interpreter identity and every installed dependency version."""
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "distributions": sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()),
    }


def native_binaries() -> list[Path]:
    """Find installed extension bytes without importing the native package."""
    spec = importlib.util.find_spec("innovate")
    if spec is None or not spec.submodule_search_locations:
        return []
    return sorted(
        path
        for directory in spec.submodule_search_locations
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (path := Path(directory) / f"_native{suffix}").is_file()
    )


def run_process(command: list[str], directory: Path, env: dict, log: Path, timeout: float) -> tuple[int, float]:
    """Bound a subprocess tree and retain output only in the ignored local cache."""
    start = time.perf_counter()
    options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    with log.open("w", encoding="utf-8") as output:
        with subprocess.Popen(
            command, cwd=directory, env=env, stdout=output, stderr=subprocess.STDOUT, **options
        ) as process:

            def stop_tree():
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                            capture_output=True,
                            check=False,
                            timeout=10,
                        )
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()

            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_tree()
                code = 124
            except BaseException:
                stop_tree()
                raise
    return code, time.perf_counter() - start


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_tools(mode: str) -> dict:
    names = ["pytest"]
    if mode != "baseline":
        names.append("coverage")
    if mode in ("testmon", "gremlins"):
        names.append(f"pytest-{'testmon' if mode == 'testmon' else 'gremlins'}")
    actual = {}
    for name in names:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProfileError(f"Install the optional pinned requirements first; missing {name}") from exc
        if actual[name] != VERSIONS[name]:
            raise ProfileError(
                f"Profile qualified with {name}=={VERSIONS[name]}, found {actual[name]}; requalify before changing"
            )
    return actual


def _coverage_result(run: Path, env: dict, expected: str, timeout: int) -> dict:
    debug = (run / "core.log").read_text() if (run / "core.log").exists() else ""
    observed = set(re.findall(r"^core.py: Using core=(\w+)\s*$", debug, re.MULTILINE))
    if observed != {expected}:
        raise ProfileError(f"Coverage did not prove the requested {expected} core; observed {sorted(observed)}")
    command = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "--rcfile",
        str(run / "coverage.ini"),
        "-o",
        str(run / "coverage.json"),
    ]
    code, _ = run_process(command, ROOT, env, run / "coverage-report.log", timeout)
    report = _read_json(run / "coverage.json")
    if code or not report or not report["meta"]["branch_coverage"] or not report["files"]:
        raise ProfileError("Coverage branch report is missing or invalid")
    files = report["files"]
    if len(files) != 1 or Path(next(iter(files))).name != "check_migration.py":
        raise ProfileError("Coverage report does not match the bounded policy target")
    measured = next(iter(files.values()))
    signature = {
        key: measured[key]
        for key in ("executed_lines", "missing_lines", "excluded_lines", "executed_branches", "missing_branches")
    }
    return {
        "actual_core": expected,
        "target": "scripts/check_migration.py",
        "summary": measured["summary"],
        "measurement_sha256": _digest(signature),
    }


def _mutation_result(run: Path) -> dict:
    report = _read_json(run / "coverage/gremlins/gremlins.json")
    if not report or report.get("skipped_files") or report["summary"]["total"] <= 0:
        raise ProfileError("Mutation trial produced no complete target report")
    summary = report["summary"]
    outcomes = ("zapped", "survived", "timeout", "error", "pardoned")
    if sum(summary[key] for key in outcomes) != summary["total"] or len(report["results"]) != summary["total"]:
        raise ProfileError("Mutation outcome counts are inconsistent")
    # Survivors are useful trial results, not successful mutation qualification.
    return {
        "target": "scripts/check_migration.py",
        "summary": summary,
        "all_killed": summary["zapped"] == summary["total"],
    }


def run_profile(mode: str, scope: str, timeout: int, rebuild: bool) -> tuple[dict, int]:
    """Execute a serial profile and emit an honest source-bound local receipt."""
    started = time.perf_counter()
    validate_environment(dict(os.environ), sys.version_info[:3])
    tools = _require_tools(mode)
    if mode in ("coverage-ctrace", "coverage-sysmon", "gremlins") and scope != "policy":
        raise ProfileError("Coverage and mutation are bounded to --scope policy")
    cache = ROOT / ".pytest_cache/test_acceleration"
    cache.mkdir(parents=True, exist_ok=True)
    with cache_lease(cache):
        run = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=cache))
        state_dir = cache / f"testmon-{scope}"
        state_dir.mkdir(exist_ok=True)
        state_file = state_dir / "state.json"
        state = _read_json(state_file) or {}
        before = snapshot(ROOT, scope, runtime_inventory(), native_paths=native_binaries())
        reasons = invalidation_reasons(state.get("snapshot"), before) if mode == "testmon" else []
        database = state_dir / ".testmondata"
        if mode == "testmon" and not database.is_file():
            reasons.append("missing_database")
        if mode == "testmon" and rebuild:
            reasons.append("explicit_rebuild")
        if reasons:
            for suffix in ("", "-shm", "-wal", "-journal"):
                database.with_name(database.name + suffix).unlink(missing_ok=True)
        if mode == "testmon":
            # A timeout, interrupt or unexpected error must not leave a reusable
            # success stamp beside a partially updated database.
            state_file.unlink(missing_ok=True)
        env = {
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(ROOT),
            "INNOVATE_TEST_METRICS": str(run / "counts.json"),
        }
        pytest_args = [
            "-c",
            str(ROOT / "pyproject.toml"),
            "-o",
            "addopts=-ra -q --tb=short",
            "-p",
            "scripts.test_acceleration.metrics",
        ]
        pytest_args.extend(str(ROOT / path) for path in SCOPES[scope])
        command = [sys.executable, "-m", "pytest", *pytest_args]
        directory = ROOT
        staged_inputs = {}
        if mode == "testmon":
            env["TESTMON_DATAFILE"] = str(database)
            command.extend(["-p", "testmon.pytest_testmon", "--testmon"])
            if reasons:
                command.append("--testmon-noselect")
        elif mode.startswith("coverage-"):
            core = mode.removeprefix("coverage-")
            (run / "coverage.ini").write_text("[run]\nbranch = True\nconcurrency = thread\n")
            env.update(
                COVERAGE_CORE=core, COVERAGE_FILE=str(run / ".coverage"), COVERAGE_DEBUG_FILE=str(run / "core.log")
            )
            command = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(run / "coverage.ini"),
                "--branch",
                "--debug=core",
                f"--include={ROOT / 'scripts/check_migration.py'}",
                "-m",
                "pytest",
                *pytest_args,
            ]
        elif mode == "gremlins":
            directory = run
            # Gremlins 1.13.2 fixes its JSON path under pytest's rootdir. Stage
            # the actual policy, tests and package/configuration markers so both
            # parent and mutant subprocesses import the same instrumented bytes.
            for name in (
                "scripts/__init__.py",
                "scripts/check_migration.py",
                "tests/native/test_migration_mutation.py",
                "migration-ledger.json",
                "pyproject.toml",
            ):
                target = run / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
                staged_inputs[name] = _file_digest(target)
            (run / "_innovate_test_metrics.py").write_bytes(
                (ROOT / "scripts/test_acceleration/metrics.py").read_bytes()
            )
            env["PYTHONPATH"] = str(run)
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(run / "pyproject.toml"),
                f"--rootdir={run}",
                "-o",
                "addopts=-ra -q --tb=short",
                "-p",
                "_innovate_test_metrics",
                "-p",
                "pytest_gremlins.plugin",
                str(run / "tests/native/test_migration_mutation.py"),
                "--gremlins",
                f"--gremlin-targets={run / 'scripts/check_migration.py'}",
                "--gremlin-operators=comparison,boolean",
                "--gremlin-coverage-timeout=60",
                "--gremlin-mutant-timeout=10",
                "--gremlin-report=json",
            ]
        code, wall = run_process(command, directory, env, run / "pytest.log", timeout)
        counts = _read_json(run / "counts.json")
        status = classify_result(
            mode, code, counts, cached_items=state.get("inventory_count", 0), rebuilt=bool(reasons)
        )
        receipt = {
            "schema_version": 1,
            "profile": mode,
            "scope": scope,
            "python": platform.python_version(),
            "tools": tools,
            "snapshot": before,
            "pytest_wall_seconds": wall,
            "pytest": counts,
            "process_exit_code": code,
            "status": status,
            "invalidation_reasons": reasons,
            "seed_inventory_count": state.get("inventory_count", 0) if mode == "testmon" and not reasons else None,
            "promotion_evidence": False,
        }
        try:
            if status == "passed" and mode.startswith("coverage-"):
                receipt["coverage"] = _coverage_result(run, env, mode.removeprefix("coverage-"), timeout)
            elif status == "passed" and mode == "gremlins":
                receipt["mutation"] = _mutation_result(run)
                receipt["mutation"]["staged_inputs_sha256"] = staged_inputs
                if not receipt["mutation"]["all_killed"]:
                    receipt["status"] = "mutation_incomplete"
                if any(_file_digest(run / name) != digest for name, digest in staged_inputs.items()):
                    raise ProfileError("Mutation trial changed a staged input")
            if before != snapshot(ROOT, scope, runtime_inventory(), native_paths=native_binaries()):
                receipt["status"] = "inputs_changed_during_run"
        except ProfileError as exc:
            receipt.update(status="failed", diagnostic=str(exc))
        if mode == "testmon":
            state_file.unlink(missing_ok=True)
            if receipt["status"] in ("passed", "no_tests_selected") and database.is_file():
                inventory_count = counts["collected"] if reasons else state.get("inventory_count", 0)
                _write_json(state_file, {"snapshot": before, "inventory_count": inventory_count})
        receipt["wall_seconds"] = time.perf_counter() - started
        _write_json(run / "receipt.json", receipt)
        receipt["local_receipt"] = str((run / "receipt.json").relative_to(ROOT))
        return receipt, 0 if receipt["status"] in ("passed", "no_tests_selected") else 1


def main(argv=None) -> int:
    """Run the explicitly selected local profile without passthrough pytest flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("baseline", "testmon", "coverage-ctrace", "coverage-sysmon", "gremlins"))
    parser.add_argument("--scope", choices=SCOPES)
    parser.add_argument(
        "--rebuild", action="store_true", help="Discard and fully seed only this profile's testmon database"
    )
    parser.add_argument("--timeout", type=int, default=180, help="Whole pytest process-tree budget, 1–300 seconds")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 300:
        parser.error("timeout must be between 1 and 300 seconds")
    if args.rebuild and args.profile != "testmon":
        parser.error("--rebuild is only valid for testmon")
    scope = args.scope or ("native" if args.profile in ("baseline", "testmon") else "policy")
    try:
        receipt, code = run_profile(args.profile, scope, args.timeout, args.rebuild)
    except (ProfileError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Local test profile refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return code
