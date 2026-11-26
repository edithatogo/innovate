"""Replay the frozen analytic inputs through the native Python binding."""

import json
from pathlib import Path

import polars as pl
import pytest

import innovate

MANIFEST = Path(__file__).resolve().parents[2] / "validation/oracles/manifest.json"
CASES = json.loads(MANIFEST.read_bytes())["oracles"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["model"])
def test_frozen_prediction_matches_analytic_oracle(case):
    parameters = case["input"].copy()
    frame = pl.DataFrame({"time": [parameters.pop("t")]})
    tolerance = case["tolerance"]
    model = case["model"]
    if model == "norton_bass":
        launch = parameters.pop("tau2")
        options = {"n_generations": 2, "variant": "successive_generations", "launch_times": [0.0, launch]}
        prediction = innovate.predict(model, frame, parameters, model_options=options)
        for generation in (1, 2):
            expected = case[f"expected_gen{generation}_before_launch"]
            assert prediction[f"prediction_{generation}"].item() == pytest.approx(expected, abs=tolerance, rel=0.0)
    else:
        prediction = innovate.predict(model, frame, parameters)
        assert prediction["prediction"].item() == pytest.approx(case["expected"], abs=tolerance, rel=0.0)
