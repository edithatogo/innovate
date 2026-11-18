"""Validate optional native notebooks and, explicitly, execute/export them.

The default gate needs only the standard library. --execute requires the locked
notebook environment, runs the real native calculation and records its evidence.
Neither mode establishes scientific validity or browser/WASM compatibility.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("docs/contracts/learning-notebooks.json")
REPORT = Path("docs/astro-site/.reports/notebooks/qualification.json")


def _source(root: Path, name: object) -> Path:
    """Resolve an ordinary repository file without traversal or symlink aliases."""
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or any(c in name for c in "\\?#\0")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"unsafe learning source: {name!r}")
    path = root
    for part in name.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlinked learning source: {name}")
    if not path.is_file():
        raise ValueError(f"missing learning source: {name}")
    return path


def validate(root: Path, contract: dict) -> dict[str, str]:
    """Validate declared notebook sources, environment pins and dataset integrity.

    Returns:
        SHA-256 fingerprints of the inspected source and data files.

    Raises:
        ValueError: If declarations, paths, source syntax or dataset checksums fail.
        OSError: If a required local file cannot be read.
    """
    keys = {
        "schema_version",
        "environment",
        "execution_mode",
        "browser_wasm_qualified",
        "static_export_published",
        "notebooks",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != keys
        or type(contract["schema_version"]) is not int
        or contract["schema_version"] != 1
        or contract["execution_mode"] != "native_python"
        or contract["browser_wasm_qualified"] is not False
        or contract["static_export_published"] is not False
    ):
        raise ValueError("unsupported notebook contract or unqualified delivery claim")
    env = contract["environment"]
    if (
        not isinstance(env, dict)
        or set(env) != {"python", "marimo", "polars", "innovate", "project", "lock"}
        or any(
            not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+)?", str(env[k]))
            for k in ("python", "marimo", "polars", "innovate")
        )
        or env["project"] != "notebooks/pyproject.toml"
        or env["lock"] != "notebooks/uv.lock"
    ):
        raise ValueError("invalid optional notebook environment")
    paths = {
        _source(root, name)
        for name in (
            env["project"],
            env["lock"],
            str(CONTRACT),
            "pyproject.toml",
            "docs/contracts/content-quality.json",
            "scripts/check_learning_notebooks.py",
        )
    }
    project = tomllib.loads((root / env["project"]).read_text())["project"]
    core = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    locked = tomllib.loads((root / env["lock"]).read_text())["package"]
    if f"marimo=={env['marimo']}" not in project["dependencies"]:
        raise ValueError("marimo project pin differs from the contract")
    if (
        project["requires-python"] != core["requires-python"]
        or f">={env['python']}" not in project["requires-python"]
        or core["version"] != env["innovate"]
        or f"polars=={env['polars']}" not in core["dependencies"]
    ):
        raise ValueError("notebook and core environment requirements differ from the contract")
    for package in ("marimo", "polars", "innovate"):
        if [item["version"] for item in locked if item["name"] == package] != [env[package]]:
            raise ValueError(f"locked {package} version differs from the contract")
    content = json.loads((root / "docs/contracts/content-quality.json").read_text())
    entries = contract["notebooks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("notebook inventory cannot be empty")
    seen: set[str] = set()
    declared: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "id",
                "source",
                "example",
                "dataset",
                "default_cutoff",
                "alternate_cutoff",
                "result_variable",
                "control_variable",
            }
            or not re.fullmatch(r"[a-z][a-z0-9-]+", str(entry["id"]))
            or entry["id"] in seen
        ):
            raise ValueError("invalid or duplicate notebook entry")
        seen.add(entry["id"])
        if any(not isinstance(entry[key], str) for key in ("source", "example", "dataset")):
            raise ValueError("notebook source, example and dataset must be repository paths")
        if (
            entry["source"] in declared
            or not entry["source"].startswith("notebooks/")
            or not entry["source"].endswith(".py")
            or entry["example"] not in content["examples"]
            or content["examples"][entry["example"]]["status"] != "runnable"
        ):
            raise ValueError("notebook must bind a unique source to a registered runnable example")
        declared.add(entry["source"])
        for key in ("default_cutoff", "alternate_cutoff"):
            if type(entry[key]) is not int or not 2010 <= entry[key] <= 2020:
                raise ValueError("cutoff must use the public-data workflow's supported years")
        if entry["default_cutoff"] == entry["alternate_cutoff"]:
            raise ValueError("changed-control qualification needs a distinct cutoff")
        if entry["result_variable"] != "comparison" or entry["control_variable"] != "cutoff":
            raise ValueError("the execution contract must inspect the declared calculation and control")
        for key in ("source", "example", "dataset"):
            paths.add(_source(root, entry[key]))
        try:
            ast.parse((root / entry["source"]).read_text())
            ast.parse((root / entry["example"]).read_text())
        except SyntaxError as error:
            raise ValueError(f"invalid learning Python source: {error}") from error
        dataset = json.loads((root / entry["dataset"]).read_text())
        if (
            not isinstance(dataset, dict)
            or type(dataset["schema_version"]) is not int
            or dataset["schema_version"] != 1
            or dataset["license"] != "CC-BY-4.0"
            or dataset["time_unit"] != "year"
            or dataset["unit"] != "percent_of_population"
            or not dataset["attribution"]
            or not dataset["observation"]
            or not dataset["limitations"]
            or set(dataset["files"]) != {"raw.json", "observations.csv"}
        ):
            raise ValueError("public dataset semantics, attribution and files are required")
        for name, digest in dataset["files"].items():
            path = _source(root, (Path(entry["dataset"]).parent / name).as_posix())
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"public dataset checksum mismatch: {name}")
            paths.add(path)
    actual = {p.relative_to(root).as_posix() for p in (root / "notebooks").glob("*.py")}
    if declared != actual:
        raise ValueError("notebook source inventory is incomplete or stale")
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_export(returncode: int, output: Path) -> str:
    """Require a successful exporter and a nonempty HTML artifact.

    Returns:
        SHA-256 of the exported artifact, for this run only.

    Raises:
        ValueError: If execution failed even when an HTML file was written.
    """
    if returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise ValueError("marimo export failed; an HTML file alone is not successful execution")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _module(path: Path, name: str):
    """Load an explicit local source under a private qualification module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load learning module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(root: Path, contract: dict, output: Path) -> dict:
    """Execute default and changed notebook controls against the ordinary example."""
    import innovate
    import innovate._native as native

    env = contract["environment"]
    actual = {name: importlib.metadata.version(name) for name in ("marimo", "polars", "innovate")}
    actual["python"] = ".".join(map(str, sys.version_info[:3]))
    if any(actual[name] != env[name] for name in actual):
        raise ValueError(f"notebook environment differs from its contract: {actual}")
    if Path(innovate.__file__).resolve() != (root / "bindings/python/innovate/__init__.py").resolve():
        raise ValueError("notebook execution must use the current checkout's installed binding")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))  # Make the repository's examples namespace importable.
    runs = []
    try:
        for entry in contract["notebooks"]:
            subprocess.run(
                [sys.executable, "-m", "marimo", "check", "--strict", entry["source"]],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            notebook = _module(root / entry["source"], "_qualified_notebook")
            example = _module(root / entry["example"], "_qualified_example")
            comparisons = []
            for cutoff in (entry["default_cutoff"], entry["alternate_cutoff"]):
                overrides = (
                    None
                    if cutoff == entry["default_cutoff"]
                    else {entry["control_variable"]: SimpleNamespace(value=cutoff)}
                )
                _, definitions = notebook.app.run(defs=overrides)
                comparison = definitions[entry["result_variable"]]
                if (
                    comparison != example.run_comparison(cutoff)
                    or not comparison["predictions"]
                    or comparison["cutoff_year"] != cutoff
                ):
                    raise ValueError("notebook results differ from the actual ordinary calculation")
                comparisons.append(comparison)
            if comparisons[0]["predictions"] == comparisons[1]["predictions"]:
                raise ValueError("changed control did not change the notebook calculation")
            exported = output / f"{entry['id']}.html"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "html",
                    entry["source"],
                    "--output",
                    str(exported),
                    "--force",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            export_hash = verify_export(result.returncode, exported)
            runs.append(
                {
                    "id": entry["id"],
                    "cutoffs": [c["cutoff_year"] for c in comparisons],
                    "holdout_rows": [c["holdout_rows"] for c in comparisons],
                    "default_scores": comparisons[0]["scores"],
                    "exact_example_parity": True,
                    "export_sha256": export_hash,
                    "export_bytes": exported.stat().st_size,
                    "export_exit_code": result.returncode,
                }
            )
    finally:
        sys.path.pop(0)
    return {
        "environment": actual,
        "native_binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "native_binary_source_binding_verified": False,
        "runs": runs,
    }


def main() -> int:
    """Write a bounded source or native execution receipt and fail on any error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "executed": False,
        "browser_wasm_qualified": False,
        "static_export_published": False,
        "scientific_acceptance": False,
    }
    try:
        contract = json.loads((ROOT / CONTRACT).read_text())
        report["source_files"] = validate(ROOT, contract)
        if args.execute:
            report.update(execute(ROOT, contract, (ROOT / REPORT).parent))
            report["executed"] = True
        report["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, ImportError, AssertionError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    destination = ROOT / REPORT if args.execute else (ROOT / REPORT).with_name("structure.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(report, indent=2)
        if args.json
        else f"Learning notebooks: {report['status']}; native execution: {report['executed']}; report: {destination}"
    )
    if "error" in report and not args.json:
        print(report["error"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
