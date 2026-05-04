"""Contract tests for the voiage diffusion-uncertainty ecosystem fixture."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

FIXTURE_ROOT = Path("specs/ecosystem/voiage/uncertainty/diffusion_v1")
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _rows(relative_path: str) -> list[dict[str, str]]:
    with (FIXTURE_ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_voiage_uncertainty_fixture_manifest_declares_portable_contract() -> None:
    """The manifest should define a versioned, dependency-free VOI fixture."""
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["fixture_id"] == "voiage_diffusion_uncertainty_v1"
    assert manifest["artifact_type"] == "diffusion_uncertainty_fixture"
    assert manifest["promotion_stage"] == "documented"
    assert manifest["consumer"] == "voiage"
    assert manifest["requires_voiage_runtime"] is False
    assert "voiage" not in manifest["base_dependencies"]
    assert manifest["payload_encoding"] == "csv-with-arrow-compatible-schema"


def test_voiage_uncertainty_fixture_maps_dimensions_to_voi_concepts() -> None:
    """Fixture metadata should identify how uncertainty dimensions feed VOI examples."""
    manifest = _manifest()
    dimensions = manifest["uncertainty_dimensions"]
    mapping = manifest["voi_concept_mapping"]

    assert dimensions == {
        "scenario_id": "decision or implementation scenario",
        "draw_id": "joint parameter uncertainty sample",
        "uncertainty_label": "uncertainty source label",
        "time": "diffusion trajectory time point",
    }
    assert mapping["draw_id"] == "Monte Carlo sample for EVPI, EVPPI, EVSI, and ENBS examples"
    assert mapping["parameter_name"] == "candidate EVPPI grouping variable"
    assert mapping["adoption"] == "decision-relevant uncertain outcome"
    assert "VOI method implementation remains outside innovate" in manifest["out_of_scope"]


def test_voiage_uncertainty_fixture_tables_have_stable_dimensions_and_columns() -> None:
    """The compact CSV payloads should match the manifest dimensions."""
    manifest = _manifest()
    dimensions = manifest["sample_dimensions"]
    parameter_rows = _rows("parameter_draws.csv")
    trajectory_rows = _rows("adoption_trajectories.csv")

    assert dimensions == {
        "scenario_count": 2,
        "draw_count_per_scenario": 3,
        "parameter_count_per_draw": 3,
        "time_point_count_per_draw": 4,
        "trajectory_row_count": 24,
        "parameter_draw_row_count": 18,
    }
    assert len(parameter_rows) == dimensions["parameter_draw_row_count"]
    assert len(trajectory_rows) == dimensions["trajectory_row_count"]

    assert set(parameter_rows[0]) == {
        "schema_version",
        "scenario_id",
        "uncertainty_label",
        "draw_id",
        "parameter_name",
        "value",
        "distribution",
        "unit",
    }
    assert set(trajectory_rows[0]) == {
        "schema_version",
        "scenario_id",
        "intervention_id",
        "uncertainty_label",
        "draw_id",
        "time",
        "adoption",
        "cumulative_adoption",
        "population",
        "segment",
    }


def test_voiage_uncertainty_fixture_values_are_deterministic_and_joinable() -> None:
    """Parameter and trajectory rows should be deterministic and joinable by scenario and draw."""
    parameter_rows = _rows("parameter_draws.csv")
    trajectory_rows = _rows("adoption_trajectories.csv")

    parameter_keys = {(row["scenario_id"], row["draw_id"]) for row in parameter_rows}
    trajectory_keys = {(row["scenario_id"], row["draw_id"]) for row in trajectory_rows}
    assert trajectory_keys == parameter_keys

    assert sum(float(row["value"]) for row in parameter_rows if row["parameter_name"] == "p") == pytest.approx(0.174)
    assert sum(float(row["adoption"]) for row in trajectory_rows if row["time"] == "3") == pytest.approx(0.852)
    assert all(row["schema_version"] == "1.0" for row in parameter_rows + trajectory_rows)


def test_voiage_uncertainty_fixture_is_documented_from_ecosystem_docs() -> None:
    """Ecosystem docs should link the fixture and keep VOI methods out of scope."""
    contract = Path("specs/ecosystem/README.md").read_text(encoding="utf-8")
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text(encoding="utf-8")
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text(encoding="utf-8")

    assert "voiage/uncertainty/diffusion_v1/manifest.json" in contract
    assert "diffusion-uncertainty fixture" in strategy
    assert "VOI method implementation remains owned outside" in strategy
    assert "This fixture is a decision-relevant uncertainty source" in strategy
    assert "specs/ecosystem/voiage/uncertainty/diffusion_v1/manifest.json" in roadmap
