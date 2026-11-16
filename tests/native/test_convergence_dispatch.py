"""Convergence dispatch, dependency closure, and Python-absent CLI tests (R05/R06/R07)."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import polars as pl
import pytest

import innovate


def test_python_binding_dispatches_natively_without_pandas_or_xarray():
    """R05/R06: Verify native Polars dispatch and absence of pandas/xarray in core deps/source."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "polars" in deps
    assert "pandas" not in deps
    assert "xarray" not in deps

    binding_src = Path("bindings/python/innovate/__init__.py").read_text(encoding="utf-8")
    assert "import pandas" not in binding_src
    assert "import xarray" not in binding_src

    df = pl.DataFrame({"time": [0.0, 1.0, 2.0, 3.0, 4.0], "observed": [0.0, 10.0, 30.0, 60.0, 85.0]})
    fit = innovate.fit("bass", df)
    assert fit.envelope["metadata"]["runtime"] == "rust_native"

    with pytest.raises(TypeError):
        innovate.fit("bass", {"not": "a_dataframe"})  # type: ignore[arg-type]


def _native_cli_from_cargo_output(output: str) -> Path:
    """Select the executable reported by Cargo, irrespective of target directory or OS."""
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue  # A build tool or procedural macro may also write to stdout.
        if not isinstance(message, dict):
            continue
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == "innovate-native"
            and "bin" in target.get("kind", [])
            and message.get("executable")
        ):
            return Path(message["executable"])
    raise AssertionError("Cargo did not report the innovate-native executable")


@pytest.mark.parametrize("name", ["innovate-native", "innovate-native.exe"])
def test_native_cli_artifact_is_selected_from_cargo_output(tmp_path: Path, name: str):
    executable = tmp_path / "custom-target" / "debug" / name
    output = "build output outside Cargo's JSON stream\n" + "\n".join(
        json.dumps(message)
        for message in [
            {"reason": "compiler-artifact", "target": {"name": "other", "kind": ["bin"]}, "executable": "wrong"},
            {
                "reason": "compiler-artifact",
                "target": {"name": "innovate-native", "kind": ["bin"]},
                "executable": str(executable),
            },
        ]
    )
    assert _native_cli_from_cargo_output(output) == executable
    with pytest.raises(AssertionError, match="Cargo did not report"):
        _native_cli_from_cargo_output('{"reason": "build-finished", "success": true}')


def test_innovate_native_cli_executes_without_python_on_path(tmp_path: Path):
    """R07: Build the locked native CLI and execute it without Python on PATH."""
    build = subprocess.run(
        ["cargo", "build", "--locked", "-p", "innovate-core", "--bin", "innovate-native", "--message-format=json"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    binary = _native_cli_from_cargo_output(build.stdout)
    assert binary.is_file(), f"missing compiled binary at {binary}"
    child_env = {"PATH": ""}
    if "SystemRoot" in os.environ:
        child_env["SystemRoot"] = os.environ["SystemRoot"]

    req_file = tmp_path / "predict_req.json"
    out_file = tmp_path / "predict_out.json"
    req_file.write_text(
        json.dumps(
            {
                "model_key": "bass",
                "payload": {
                    "inputs": {"time": [0.0, 1.0, 2.0]},
                    "parameters": {"p": 0.03, "q": 0.38, "m": 1000.0},
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(binary), "predict", "--request", str(req_file), "--output", str(out_file)],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    resp = json.loads(out_file.read_text(encoding="utf-8"))
    assert resp.get("error") is None
    assert len(resp["result"]["values"]) == 3

    fit_req = tmp_path / "fit_req.json"
    fit_out = tmp_path / "fit_out.json"
    fit_req.write_text(
        json.dumps(
            {
                "model_key": "bass",
                "payload": {
                    "inputs": {
                        "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                        "observed": [0.0, 10.0, 30.0, 60.0, 85.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    fit_proc = subprocess.run(
        [str(binary), "fit", "--request", str(fit_req), "--output", str(fit_out)],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert fit_proc.returncode == 0, fit_proc.stderr
    fit_resp = json.loads(fit_out.read_text(encoding="utf-8"))
    assert fit_resp.get("error") is None
    assert "p" in fit_resp["result"]["parameters"]

    bad_req = tmp_path / "bad_req.json"
    bad_out = tmp_path / "bad_out.json"
    bad_req.write_text(
        json.dumps({"model_key": "nonexistent_model", "payload": {}}),
        encoding="utf-8",
    )
    bad_proc = subprocess.run(
        [str(binary), "predict", "--request", str(bad_req), "--output", str(bad_out)],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_proc.returncode != 0
