"""Validate and classify documentation examples and API snippets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/source/_static/astro_starlight/example_validation.json"


PYTHON_SMOKE = """
from innovate import BassModel

model = BassModel()
values = model.cumulative_adoption([0, 1, 2], 0.03, 0.38, 1000)
assert len(values) == 3
assert all(value >= 0 for value in values)
"""


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _run_python_smoke() -> dict[str, Any]:
    command = [sys.executable, "-c", PYTHON_SMOKE]
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    return {
        "id": "python_api_smoke",
        "language": "python",
        "source_path": "docs/astro-site/src/content/docs/api/python.md",
        "status": "runnable" if completed.returncode == 0 else "failed",
        "classification": "runnable_in_current_environment",
        "command": "uv run python -c '<BassModel API smoke>'",
        "returncode": completed.returncode,
    }


def build_evidence() -> dict[str, Any]:
    """Build example validation evidence."""
    evidence_date = date.today().isoformat()
    examples = [
        _run_python_smoke(),
        {
            "id": "r_binding_end_to_end",
            "language": "r",
            "source_path": _rel(ROOT / "bindings/r/inst/examples/end_to_end.R"),
            "status": "classified",
            "classification": "runnable_in_language_ci",
            "command": "Rscript bindings/r/tests/run.R",
        },
        {
            "id": "julia_binding_end_to_end",
            "language": "julia",
            "source_path": _rel(ROOT / "bindings/julia/examples/end_to_end.jl"),
            "status": "classified",
            "classification": "runnable_in_language_ci",
            "command": "julia --project=bindings/julia bindings/julia/test/runtests.jl",
        },
        {
            "id": "typescript_diagnostics_workflow",
            "language": "typescript",
            "source_path": _rel(ROOT / "bindings/typescript/examples/diagnostics-workflow.ts"),
            "status": "classified",
            "classification": "runnable_in_language_ci",
            "command": "npm test --prefix bindings/typescript",
        },
        {
            "id": "go_binding_example_test",
            "language": "go",
            "source_path": _rel(ROOT / "bindings/go/example_test.go"),
            "status": "classified",
            "classification": "runnable_in_language_ci",
            "command": "go test ./...",
        },
        {
            "id": "rust_memory_profile_example",
            "language": "rust",
            "source_path": _rel(ROOT / "bindings/rust/examples/profile_memory_native_kernels.rs"),
            "status": "classified",
            "classification": "optional_dependency_or_toolchain",
            "command": "cargo check --example profile_memory_native_kernels",
        },
    ]
    overall_status = "passed" if all(example["status"] != "failed" for example in examples) else "failed"
    return {
        "schema_version": 1,
        "generated_by_track": "production_docs_observability_20260614",
        "generated_at": f"{evidence_date}T00:00:00Z",
        "evidence_date": evidence_date,
        "overall_status": overall_status,
        "ci_evidence": {
            "nox_session": "examples",
            "command": "uv run nox -s examples",
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print validation JSON to stdout.")
    args = parser.parse_args()

    evidence = build_evidence()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
