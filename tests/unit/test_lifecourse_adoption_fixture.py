"""Contract checks for the lifecourse adoption-trajectory fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

FIXTURE_ROOT = Path("specs/ecosystem/lifecourse/adoption_trajectory/v1")
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
PAYLOAD_PATH = FIXTURE_ROOT / "adoption_trajectory.parquet"
REQUIRED_COLUMNS = {
    "scenario_id": "string",
    "intervention_id": "string",
    "time": "int32",
    "adoption": "double",
    "cumulative_adoption": "double",
    "population": "int32",
    "segment": "string",
    "uncertainty_label": "string",
}


def test_lifecourse_adoption_fixture_manifest_defines_portable_contract() -> None:
    """The manifest should define a versioned contract without private objects."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "innovate.ecosystem.adoption_trajectory.v1"
    assert manifest["fixture_id"] == "lifecourse_adoption_trajectory_smoke_v1"
    assert manifest["artifact_type"] == "adoption_curve"
    assert manifest["promotion_stage"] == "documented"
    assert manifest["runtime_adapter_status"] == "future_work"
    assert manifest["payload"]["format"] == "parquet"
    assert manifest["payload"]["path"] == PAYLOAD_PATH.name
    assert manifest["payload"]["row_count"] == 12
    assert manifest["compatibility"]["arrow_compatible"] is True
    assert manifest["compatibility"]["parquet_compatible"] is True
    assert manifest["dependency_policy"]["lifecourse_required"] is False
    assert manifest["dependency_policy"]["innovate_base_import_required"] is False
    assert manifest["producer"]["project"] == "innovate"
    assert manifest["consumer"]["project"] == "lifecourse"
    assert "private implementation classes" in manifest["consumer"]["must_not"]

    columns = {column["name"]: column for column in manifest["schema"]["columns"]}
    assert {name: column["arrow_type"] for name, column in columns.items()} == REQUIRED_COLUMNS
    assert all(not column["nullable"] for column in columns.values())


def test_lifecourse_adoption_fixture_payload_matches_manifest_schema() -> None:
    """The Parquet payload should be small, deterministic, and Arrow-readable."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    table = pq.read_table(PAYLOAD_PATH)

    assert table.num_rows == manifest["payload"]["row_count"]
    assert table.column_names == list(REQUIRED_COLUMNS)
    assert {field.name: str(field.type) for field in table.schema} == REQUIRED_COLUMNS

    rows = table.to_pylist()
    assert rows[0] == {
        "scenario_id": "base_case",
        "intervention_id": "vaccination_reminder",
        "time": 0,
        "adoption": 0.05,
        "cumulative_adoption": 0.05,
        "population": 10000,
        "segment": "adult_primary_care",
        "uncertainty_label": "deterministic",
    }
    assert rows[-1]["scenario_id"] == "implementation_push"
    assert rows[-1]["cumulative_adoption"] == 0.57

    for scenario_id in {row["scenario_id"] for row in rows}:
        scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
        for intervention_id in {row["intervention_id"] for row in scenario_rows}:
            trajectory = [row for row in scenario_rows if row["intervention_id"] == intervention_id]
            cumulative = [row["cumulative_adoption"] for row in trajectory]
            assert cumulative == sorted(cumulative)
            assert all(0.0 <= row["adoption"] <= 1.0 for row in trajectory)
            assert all(0.0 <= value <= 1.0 for value in cumulative)


def test_lifecourse_adoption_fixture_hash_and_docs_are_current() -> None:
    """Docs and manifest checksums should point at the committed fixture."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload_hash = hashlib.sha256(PAYLOAD_PATH.read_bytes()).hexdigest()
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text(encoding="utf-8")
    ecosystem_readme = Path("specs/ecosystem/README.md").read_text(encoding="utf-8")

    assert manifest["payload"]["sha256"] == payload_hash
    assert "lifecourse/adoption_trajectory/v1/manifest.json" in strategy
    assert "lifecourse/adoption_trajectory/v1/manifest.json" in ecosystem_readme
    assert "runtime adapter implementation remains future work" in strategy


def test_lifecourse_adoption_fixture_does_not_import_lifecourse() -> None:
    """Base fixture inspection should not require the sibling project."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    table = pq.read_table(PAYLOAD_PATH, columns=list(REQUIRED_COLUMNS))

    assert manifest["dependency_policy"]["lifecourse_required"] is False
    assert table.num_rows == 12
