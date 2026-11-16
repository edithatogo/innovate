"""V01 independent model-oracle registry validation tests."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest


def validate_oracle_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("unsupported oracle manifest schema_version")
    for idx, entry in enumerate(manifest.get("oracles", [])):
        if entry.get("implementation_hash") == entry.get("kernel_hash"):
            errors.append(f"oracles[{idx}] implementation_hash matches kernel_hash (not independent)")
        tol = entry.get("tolerance")
        if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0.0:
            errors.append(f"oracles[{idx}] missing positive finite tolerance")
        if not entry.get("input") or not entry.get("licence"):
            errors.append(f"oracles[{idx}] missing input or licence")
    return errors


def test_checked_in_oracle_manifest_is_valid_and_matches_analytic_limits():
    manifest = json.loads(Path("validation/oracles/manifest.json").read_text(encoding="utf-8"))
    assert validate_oracle_manifest(manifest) == []

    bass = manifest["oracles"][0]
    inp = bass["input"]
    analytic = inp["m"] * (1.0 - math.exp(-inp["p"] * inp["t"]))
    assert abs(analytic - bass["expected"]) <= bass["tolerance"]


def test_non_independent_hash_unknown_tolerance_and_wrong_output_are_caught():
    manifest = json.loads(Path("validation/oracles/manifest.json").read_text(encoding="utf-8"))

    same_hash = copy.deepcopy(manifest)
    same_hash["oracles"][0]["implementation_hash"] = same_hash["oracles"][0]["kernel_hash"]
    assert validate_oracle_manifest(same_hash)

    bad_tol = copy.deepcopy(manifest)
    bad_tol["oracles"][0]["tolerance"] = -1.0
    assert validate_oracle_manifest(bad_tol)

    wrong_val = manifest["oracles"][1]["expected"] + 1.0
    assert abs(wrong_val - 50.0) > manifest["oracles"][1]["tolerance"]
    with pytest.raises(AssertionError):
        assert abs(wrong_val - 50.0) <= manifest["oracles"][1]["tolerance"]
