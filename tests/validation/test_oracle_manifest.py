"""Validate the versioned independent analytic reference cases."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "validation/oracles/manifest.json"


def _file_sha256(path: str) -> str:
    source = ROOT / path
    if Path(path).is_absolute() or not source.resolve().is_relative_to(ROOT):
        raise ValueError("reference and kernel sources must be inside the repository")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _check_value(entry: dict[str, Any], tolerance: float) -> None:
    data = entry["input"]
    match entry["model"], entry["source"]:
        case "bass", "closed_form_exponential_limit_q_zero":
            if data["q"] != 0.0:
                raise ValueError("the Bass exponential limit requires q = 0")
            expected = -data["m"] * math.expm1(-data["p"] * data["t"])
            recorded = entry["expected"]
        case "logistic", "closed_form_inflection_midpoint":
            if data["t"] != data["x0"]:
                raise ValueError("the Logistic midpoint requires t = x0")
            expected = data["L"] / 2.0
            recorded = entry["expected"]
        case "norton_bass", "norton_bass_1987_conservation_identity":
            if data["t"] >= data["tau2"]:
                raise ValueError("this Norton–Bass case precedes generation-two launch")
            expected = 0.0
            recorded = entry["expected_gen2_before_launch"]
        case _:
            raise ValueError("unrecognised reference")
    if not math.isclose(expected, recorded, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError("recorded result disagrees with the analytic reference")


def validate_oracle_manifest(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema_version") != "1.1":
        return ["unsupported oracle manifest schema_version"]
    errors: list[str] = []
    entries = manifest.get("oracles")
    if not isinstance(entries, list) or not entries:
        return ["missing oracle cases"]
    if (
        sorted(entry.get("model", "") for entry in entries if isinstance(entry, dict))
        != ["bass", "logistic", "norton_bass"]
        or len(entries) != 3
    ):
        return ["required oracle cases must each occur once"]
    for idx, entry in enumerate(entries):
        try:
            if not entry.get("input") or not entry.get("licence"):
                raise ValueError("missing input or licence")
            tol = entry["tolerance"]
            if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0.0:
                raise ValueError("missing positive finite tolerance")
            if _file_sha256(entry["reference_path"]) == _file_sha256(entry["kernel_path"]):
                raise ValueError("reference and kernel have identical source bytes")
            data = json.dumps(entry["input"], sort_keys=True, separators=(",", ":"), allow_nan=False)
            if hashlib.sha256(data.encode()).hexdigest() != entry["input_sha256"]:
                raise ValueError("input fingerprint differs")
            _check_value(entry, tol)
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"oracles[{idx}]: {error}")
    return errors


def test_checked_in_oracle_manifest_is_valid_and_matches_analytic_limits():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert validate_oracle_manifest(manifest) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kernel_path", "tests/validation/test_oracle_manifest.py", "identical source bytes"),
        ("tolerance", -1.0, "positive finite tolerance"),
        ("input_sha256", "0" * 64, "input fingerprint differs"),
        ("expected", 1.0, "disagrees with the analytic reference"),
    ],
)
def test_invalid_independence_tolerance_inputs_and_result_are_caught(field, value, message):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(manifest)
    candidate["oracles"][0][field] = value
    assert any(message in error for error in validate_oracle_manifest(candidate))
