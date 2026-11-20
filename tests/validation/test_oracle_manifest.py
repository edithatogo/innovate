"""Verify frozen analytic references against the installed native consumer."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "validation/oracles"
KERNEL = ROOT / "rust/crates/innovate-core/src"
KERNEL_FILES = {"kernel.rs", "substitution.rs", "competition.rs"}
MODELS = ("bass", "logistic", "gompertz", "fisher_pry", "norton_bass", "multi_product")


@pytest.fixture
def manifest():
    return json.loads((ORACLES / "manifest.json").read_text(encoding="utf-8"))


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def input_sha256(inputs: dict) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_manifest(manifest: dict) -> None:
    assert manifest["schema_version"] == "1.1", "unsupported oracle manifest version"
    reference = manifest["reference"]
    assert reference["licence"] == "Apache-2.0"
    derivations = ORACLES / "derivations.md"
    assert reference["sha256"] == text_sha256(derivations), "stale reference source"
    assert set(manifest["kernel_sha256"]) == KERNEL_FILES
    for name, digest in manifest["kernel_sha256"].items():
        assert digest != reference["sha256"], "oracle is the kernel under test"
        assert digest == text_sha256(KERNEL / name), f"stale native source for {name}"
    headings = derivations.read_text(encoding="utf-8").splitlines()
    assert tuple(case["model"] for case in manifest["oracles"]) == MODELS
    for case in manifest["oracles"]:
        assert f"## {case['derivation']}" in headings, "unknown analytic reference"
        inputs = case["input"]
        assert set(inputs) == {"time", "parameters", "model_options"}
        assert case["input_sha256"] == input_sha256(inputs), "stale oracle inputs"
        tolerance = case["tolerance"]
        assert type(tolerance) in (int, float) and math.isfinite(tolerance) and tolerance > 0, "invalid tolerance"
        times = inputs["time"]
        assert len(times) >= 2 and all(left < right for left, right in pairwise(times))
        expected = case["expected"]
        assert expected and all(
            len(values) == len(times) and all(math.isfinite(value) for value in values) for values in expected.values()
        ), "missing or invalid predicted levels"


def verify_prediction(case: dict) -> None:
    import polars as pl

    import innovate

    inputs = case["input"]
    frame = pl.DataFrame({"time": inputs["time"], "row_id": list(range(len(inputs["time"])))})
    result = innovate.predict(case["model"], frame, inputs["parameters"], model_options=inputs["model_options"])
    assert isinstance(result, pl.DataFrame)
    assert result.select(frame.columns).equals(frame)
    assert result.columns == frame.columns + list(case["expected"])
    for column, values in case["expected"].items():
        assert result[column].to_list() == pytest.approx(values, abs=case["tolerance"], rel=0), column


def test_checked_in_oracle_manifest_binds_sources_and_inputs(manifest):
    verify_manifest(manifest)


@pytest.mark.parametrize("index", range(len(MODELS)), ids=MODELS)
def test_installed_native_predictions_match_independent_references(manifest, index):
    verify_manifest(manifest)
    verify_prediction(manifest["oracles"][index])


def test_changed_input_cannot_reuse_the_original_expected_values(manifest):
    manifest["oracles"][0]["input"]["time"][-1] += 1
    with pytest.raises(AssertionError, match="stale oracle inputs"):
        verify_manifest(manifest)


def test_changed_native_source_cannot_reuse_the_reviewed_fingerprint(manifest):
    manifest["kernel_sha256"]["kernel.rs"] = "0" * 64
    with pytest.raises(AssertionError, match="stale native source"):
        verify_manifest(manifest)


def test_a_kernel_with_the_oracle_fingerprint_is_rejected(manifest):
    manifest["kernel_sha256"]["kernel.rs"] = manifest["reference"]["sha256"]
    with pytest.raises(AssertionError, match="oracle is the kernel"):
        verify_manifest(manifest)


@pytest.mark.parametrize("tolerance", [-1, 0, math.inf, math.nan, True])
def test_invalid_tolerance_is_rejected(manifest, tolerance):
    manifest["oracles"][0]["tolerance"] = tolerance
    with pytest.raises(AssertionError, match="invalid tolerance"):
        verify_manifest(manifest)


def test_the_native_comparison_rejects_a_wrong_prediction(manifest):
    case = manifest["oracles"][0]
    case["expected"]["prediction"][-1] += 1
    with pytest.raises(AssertionError, match="prediction"):
        verify_prediction(case)
