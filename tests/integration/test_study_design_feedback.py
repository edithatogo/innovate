"""E03 acyclic study-design feedback tests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest


def simulate_study_feedback(design: dict[str, Any]) -> list[float]:
    spec = json.loads(Path("specs/study-design-v1.json").read_text(encoding="utf-8"))
    if design.get("schema_version") != spec["schema_version"]:
        raise ValueError("unsupported study-design schema_version")
    if design.get("study_type") not in spec["allowed_study_types"]:
        raise ValueError("unsupported study_type")
    n = int(design["sample_size"])
    if n <= 0:
        raise ValueError("sample_size must be positive")
    rng = random.Random(int(design["seed"]))
    true_effect = float(design["true_effect"])
    noise_sd = float(design["noise_sd"])
    return [rng.gauss(true_effect, noise_sd) for _ in range(n)]


def test_study_design_reproduces_and_rejects_unsupported_designs():
    design = {
        "schema_version": "1.0",
        "study_type": "rct",
        "sample_size": 25,
        "seed": 123,
        "true_effect": 0.4,
        "noise_sd": 0.1,
    }
    assert simulate_study_feedback(design) == simulate_study_feedback(design)

    with pytest.raises(ValueError, match="study_type"):
        simulate_study_feedback(dict(design, study_type="unverified_heuristic"))
