"""Convergence dispatch, dependency closure, and Python-absent CLI tests (R05/R06/R07)."""

from __future__ import annotations

import json
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


def test_innovate_native_cli_executes_without_python_on_path(tmp_path: Path):
    """R07: Execute compiled innovate-native binary with PATH stripped of Python."""
    binary = Path("target/debug/innovate-native").resolve()
    assert binary.exists(), f"missing compiled binary at {binary}"

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
        env={"PATH": ""},
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
        env={"PATH": ""},
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
        env={"PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_proc.returncode != 0
