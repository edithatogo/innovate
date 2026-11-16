"""Qualify the installed optional Arrow boundary with fixed, source-bound tests.

Run with the selected Python 3.15.2 environment after building and installing a
wheel. The build record is executor evidence, not a cryptographic attestation.
This command verifies its input/output identities, wheel RECORD and installed
bytes, then executes existing native consumers outside the source checkout.
It does not qualify a release, other platforms or the Rust minimum version.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

EXPORT_MODULE = "test_scenario_exports"
PRODUCER_MODULE = "test_python_consumer"
PRODUCERS = (
    "test_native_kairos_replay_and_engine_errors",
    "test_kairos_abm_forcing_retains_agents_and_restarts_adoption",
    "test_kairos_tick_checkpoint_resume",
    "test_paused_scenario_polars_export_preserves_checkpoint",
)
OUTCOMES = (
    "missing-both",
    "missing-error",
    "missing-result",
    "empty-outcome",
    "contradictory-outcome",
    "array-result",
    "scalar-result",
    "string-error",
    "array-error",
    "boolean-error",
)
EXPORTS = ("execution", "pair", "ensemble")
EXPECTED_CASES = (
    {(EXPORT_MODULE, f"test_scenario_export_requires_explicit_envelope_error[{export}]") for export in EXPORTS}
    | {
        (EXPORT_MODULE, f"test_scenario_export_rejects_incomplete_or_contradictory_outcomes[{outcome}-{export}]")
        for outcome in OUTCOMES
        for export in EXPORTS
    }
    | {(EXPORT_MODULE, "test_valid_scenario_exports_preserve_outcomes_order_and_nested_json")}
    | {
        (EXPORT_MODULE, f"test_optional_pyarrow_reads_and_roundtrips_native_scenario_exports[{ipc}-{field}]")
        for ipc in ("file", "stream")
        for field in ("executions", "comparisons")
    }
    | {(PRODUCER_MODULE, name) for name in PRODUCERS}
)
TEST_INPUTS = (
    "tests/native/test_scenario_exports.py",
    "tests/native/test_python_consumer.py",
    "rust/crates/innovate-core/tests/fixtures/kairos_network_checkpoint.json",
)
BUILD_ROOTS = {"Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock", "README.md", "LICENSE"}


def sha256(path: Path) -> str:
    """Hash a file without interpreting its content."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_versions(python_version: tuple[int, ...], installed: dict[str, str], expected: dict[str, str]) -> None:
    """Reject missing optional dependencies and version drift before pytest can skip."""
    if tuple(python_version[:3]) != (3, 15, 2):
        raise ValueError("Python 3.15.2 is required for this qualification")
    for name, version in expected.items():
        if installed.get(name) != version:
            raise ValueError(f"{name}: expected {version}, installed {installed.get(name)!r}")


def validate_native_loader(loader: object) -> None:
    """Require the imported native module to be an actual compiled extension."""
    if not isinstance(loader, importlib.machinery.ExtensionFileLoader):
        raise ValueError("Innovate native module must use the compiled extension loader")


def validate_junit(root: ET.Element) -> dict[str, int]:
    """Require every named case exactly once and reject every skipped/error result."""
    for outcome in ("skipped", "failure", "error"):
        if any(True for _ in root.iter(outcome)):
            raise ValueError(f"qualification contains {outcome}")
    actual: set[tuple[str, str]] = set()
    for case in root.iter("testcase"):
        key = (case.get("classname", "").split(".")[-1], case.get("name", ""))
        if key in actual:
            raise ValueError(f"duplicate qualification case: {key}")
        actual.add(key)
    if actual - EXPECTED_CASES:
        raise ValueError(f"unexpected qualification cases: {sorted(actual - EXPECTED_CASES)}")
    if EXPECTED_CASES - actual:
        raise ValueError(f"missing qualification cases: {sorted(EXPECTED_CASES - actual)}")
    return {"passed": 42, "exports": 38, "optional_arrow": 4, "native_producers": 4}


def verify_wheel(wheel: Path, package_root: Path, required_paths: list[Path]) -> dict[str, object]:
    """Compare every package member with its RECORD hash and installed bytes."""
    package_root = package_root.resolve()
    required = {"innovate/" + path.resolve().relative_to(package_root).as_posix() for path in required_paths}
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel member")
        for name in names:
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or "\\" in name:
                raise ValueError(f"unsafe wheel member: {name}")
        members = {name for name in names if name.startswith("innovate/") and not name.endswith("/")}
        if not members or not required.issubset(members):
            raise ValueError("wheel is missing a required imported module")
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("wheel must contain one RECORD")
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8"))))
        if any(len(row) != 3 for row in rows) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("invalid or duplicate RECORD entries")
        record = {row[0]: row[1:] for row in rows}
        for member in sorted(members):
            data = archive.read(member)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            if record.get(member) != [f"sha256={digest}", str(len(data))]:
                raise ValueError(f"RECORD mismatch: {member}")
            installed = (package_root / member.removeprefix("innovate/")).resolve()
            if not installed.is_relative_to(package_root):
                raise ValueError(f"unsafe installed package member: {member}")
            if not installed.is_file() or sha256(installed) != hashlib.sha256(data).hexdigest():
                raise ValueError(f"installed {member} differs from supplied wheel")
    return {"sha256": sha256(wheel), "verified_package_members": len(members)}


def verify_input_hashes(root: Path, identities: dict[str, str]) -> None:
    """Reject changed or missing source/test inputs, including path escapes."""
    root = root.resolve()
    for name, digest in identities.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source input mismatch: {name}")


def build_inputs(repository: Path) -> dict[str, str]:
    """Record all tracked Rust/Python build inputs without treating docs edits as native changes."""
    paths = subprocess.check_output(["git", "-C", str(repository), "ls-files"], text=True).splitlines()
    return {
        path: sha256(repository / path)
        for path in paths
        if path in BUILD_ROOTS or path.startswith(("rust/", "bindings/python/"))
    }


def validate_build_record(record: object, inputs: dict[str, str], wheel_hash: str) -> dict[str, object]:
    """Bind executor-reported successful build inputs and output to this qualification."""
    if not isinstance(record, dict):
        raise ValueError("build record must be an object")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported build record schema_version")
    if (
        record.get("status") != "built"
        or type(record.get("build_exit_code")) is not int
        or record["build_exit_code"] != 0
    ):
        raise ValueError("build record does not report a successful completed build")
    if record.get("wheel_sha256") != wheel_hash:
        raise ValueError("build record wheel hash mismatch")
    if record.get("source_inputs") != inputs:
        raise ValueError("build record source inputs mismatch")
    for field, length in (
        ("source_commit", 40),
        ("source_tree", 40),
        ("kairos_revision", 40),
        ("build_log_sha256", 64),
    ):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
            raise ValueError(f"build record requires a valid {field}")
    for field in ("rustc", "cargo", "command", "build_profile", "build_log"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"build record requires {field}")
    if type(record.get("build_jobs")) is not int or record["build_jobs"] < 1:
        raise ValueError("build record requires a positive integer build_jobs")
    changes = record.get("worktree_changes")
    if not isinstance(changes, list) or any(not isinstance(path, str) for path in changes):
        raise ValueError("build record requires worktree_changes")
    return record


def declared_versions(repository: Path) -> dict[str, str]:
    """Read exact runtime/test pins from the existing manifest rather than changing them."""
    manifest = tomllib.loads((repository / "pyproject.toml").read_text())
    expected = {"innovate": manifest["project"]["version"]}
    requirements = (
        manifest["project"]["dependencies"]
        + manifest["project"]["optional-dependencies"]["arrow"]
        + manifest["dependency-groups"]["dev"]
    )
    for name in ("polars", "pyarrow", "pytest"):
        pins = [item.removeprefix(name + "==") for item in requirements if item.startswith(name + "==")]
        if len(pins) != 1:
            raise ValueError(f"one exact {name} pin is required in pyproject.toml")
        expected[name] = pins[0]
    return expected


def qualify(repository: Path, wheel: Path, build_record: Path, output: Path) -> dict[str, object]:
    """Execute fixed native consumers in an isolated process and retain checked evidence."""
    qualifier_hash, build_record_hash = sha256(Path(__file__)), sha256(build_record)
    if not sys.flags.isolated or Path.cwd().resolve().is_relative_to(repository):
        raise ValueError("qualification must run with Python -I outside the source checkout")
    if any(Path(path).resolve().is_relative_to(repository) for path in sys.path if path):
        raise ValueError("source checkout is present on the isolated import path")
    expected = declared_versions(repository)
    installed = {}
    for name in expected:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    validate_versions(sys.version_info[:3], installed, expected)
    for name in ("pandas", "xarray"):
        if importlib.util.find_spec(name) is not None:
            raise ValueError(f"unsupported production dependency present: {name}")
    import innovate
    import innovate._native as native

    validate_native_loader(native.__loader__)
    package_root = Path(importlib.metadata.distribution("innovate").locate_file("innovate")).resolve()
    if package_root.is_relative_to(repository) or not package_root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Innovate must be installed in the selected environment outside the checkout")
    wheel_evidence = verify_wheel(wheel, package_root, [Path(innovate.__file__), Path(native.__file__)])
    inputs = build_inputs(repository)
    record = validate_build_record(json.loads(build_record.read_text()), inputs, wheel_evidence["sha256"])
    verify_input_hashes(build_record.parent, {record["build_log"]: record["build_log_sha256"]})
    tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", record["source_commit"] + "^{tree}"], text=True
    ).strip()
    if tree != record["source_tree"]:
        raise ValueError("build record source commit/tree mismatch")
    core_manifest = tomllib.loads((repository / "rust/crates/innovate-core/Cargo.toml").read_text())
    provider_pins = [
        value.get("rev") for name, value in core_manifest["dependencies"].items() if name.startswith("kairo-ecs-")
    ]
    if len(provider_pins) != 5 or set(provider_pins) != {record["kairos_revision"]}:
        raise ValueError("build record Kairos provider revision mismatch")
    test_inputs = {path: sha256(repository / path) for path in TEST_INPUTS}
    work = Path.cwd()
    for relative in TEST_INPUTS:
        destination = work / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    verify_input_hashes(work, test_inputs)
    config = work / "pytest.ini"
    config.write_text("[pytest]\nminversion = 9.2\n")
    junit_path = output.with_suffix(".junit.xml")
    command = [
        sys.executable,
        "-I",
        "-m",
        "pytest",
        "-c",
        str(config),
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        "-q",
        f"--junitxml={junit_path}",
        "tests/native/test_scenario_exports.py",
        *[f"tests/native/test_python_consumer.py::{name}" for name in PRODUCERS],
    ]
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(command, cwd=work, env=environment, capture_output=True, text=True, timeout=600)
    output.with_suffix(".pytest.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise ValueError(f"installed tests exited {completed.returncode}; see retained pytest log")
    result = validate_junit(ET.parse(junit_path).getroot())
    verify_input_hashes(repository, inputs)
    verify_input_hashes(repository, test_inputs)
    verify_input_hashes(work, test_inputs)
    if sha256(Path(__file__)) != qualifier_hash or sha256(build_record) != build_record_hash:
        raise ValueError("qualifier or build record changed during execution")
    if verify_wheel(wheel, package_root, [Path(innovate.__file__), Path(native.__file__)]) != wheel_evidence:
        raise ValueError("wheel changed during execution")
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "installed_scenario_arrow_and_four_native_producers",
        "source": {
            "commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "tree": subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "worktree_changes": subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"], text=True
            ).splitlines(),
            "build_inputs": inputs,
            "test_inputs": test_inputs,
        },
        "build_record_sha256": build_record_hash,
        "build_record": record,
        "build_evidence_type": "executor record with verified source and artifact hashes; not an attestation",
        "wheel": wheel_evidence,
        "native_module_sha256": sha256(Path(native.__file__)),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "versions": installed,
        },
        "imports": {"innovate": innovate.__file__, "native": native.__file__, "isolated": bool(sys.flags.isolated)},
        "qualifier_sha256": qualifier_hash,
        "command": command,
        "test_results": result,
        "junit_sha256": sha256(junit_path),
        "pytest_log_sha256": sha256(output.with_suffix(".pytest.log")),
        "limitations": ["development wheel on one platform", "no MSRV, full scientific or release acceptance"],
    }


def main() -> int:
    """Run the worker with isolated imports and write either a pass or failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--build-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repository, wheel, record, output = [
        path.resolve() for path in (args.repository, args.wheel, args.build_record, args.output)
    ]
    if any(path.exists() for path in (output, output.with_suffix(".junit.xml"), output.with_suffix(".pytest.log"))):
        parser.error("evidence output already exists; choose a new output path to preserve previous results")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        with TemporaryDirectory(prefix="innovate-arrow-installed-") as directory:
            return subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--repository",
                    str(repository),
                    "--wheel",
                    str(wheel),
                    "--build-record",
                    str(record),
                    "--output",
                    str(output),
                ],
                cwd=directory,
                check=False,
            ).returncode
    try:
        result = qualify(repository, wheel, record, output)
    except (ValueError, OSError, KeyError, ImportError, BadZipFile, subprocess.SubprocessError, ET.ParseError) as error:
        result = {"schema_version": 1, "status": "failed", "error": str(error)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "error": result.get("error")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
