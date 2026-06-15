"""Validation tests for advanced workflow fixtures."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = Path("tests/fixtures/advanced_runtime/workflows.json")
ASSUMPTIONS_PATH = Path("tests/fixtures/advanced_runtime/README.md")


def test_advanced_workflow_fixture_bundle_exists_and_covers_workflows() -> None:
    """The fixture bundle should cover every advanced workflow in this track."""
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "advanced_workflow_fixtures.v1"
    assert payload["reproducibility"] == {
        "seed": 20260614,
        "generated_by": "conductor advanced modeling runtime track",
        "license": "Apache-2.0",
    }
    workflows = {case["workflow"] for case in payload["cases"]}
    assert workflows == {
        "regime_ensemble",
        "policy_scenario",
        "streaming_update",
        "uncertainty_calibration",
    }


def test_advanced_workflow_fixtures_are_shape_consistent() -> None:
    """Fixture cases should have aligned time, observed, and covariate shapes."""
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in payload["cases"]:
        time = case["time"]
        observed = case["observed"]
        assert len(time) == len(observed)
        assert time == sorted(time)
        assert observed == sorted(observed)
        assert case["assumptions"]
        for values in case.get("covariates", {}).values():
            assert len(values) == len(time)


def test_advanced_workflow_fixture_assumptions_are_documented() -> None:
    """The fixture assumptions should be documented near the machine fixture."""
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")

    for expected in [
        "Regime ensemble",
        "Policy scenario",
        "Streaming update",
        "Uncertainty calibration",
        "seed 20260614",
    ]:
        assert expected in text
