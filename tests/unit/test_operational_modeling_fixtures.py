"""Tests for operational-modeling ecosystem fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path("specs/ecosystem/operational_modeling")
TREEAGE_MANIFEST = FIXTURE_ROOT / "treeage_style" / "manifest.json"
DES_MANIFEST = FIXTURE_ROOT / "des" / "manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_treeage_style_manifest_defines_decision_model_contract() -> None:
    """Decision-model fixtures should expose portable TreeAge-style metadata."""
    manifest = _load_json(TREEAGE_MANIFEST)

    assert manifest["schema_version"] == "0.1.0"
    assert manifest["artifact_type"] == "operational_modeling.treeage_style"
    assert manifest["proprietary_parsing"] == "out_of_scope"
    assert manifest["engine_dependency"] == "none"

    required_sections = {
        "decision_tree",
        "state_transition_model",
        "strategies",
        "states",
        "transitions",
        "payoffs",
        "provenance",
        "xla",
    }
    assert required_sections <= manifest.keys()

    decision_tree = manifest["decision_tree"]
    assert {"model_id", "root_node_id", "nodes", "edges"} <= decision_tree.keys()
    assert len(decision_tree["nodes"]) >= 3
    assert len(decision_tree["edges"]) >= 2

    state_transition = manifest["state_transition_model"]
    assert state_transition["cycle_length"] == "1 year"
    assert state_transition["time_horizon_cycles"] == 3

    for strategy in manifest["strategies"]:
        assert {"strategy_id", "label", "description"} <= strategy.keys()

    for state in manifest["states"]:
        assert {"state_id", "label", "absorbing"} <= state.keys()

    for transition in manifest["transitions"]:
        assert {
            "strategy_id",
            "from_state",
            "to_state",
            "cycle",
            "probability",
        } <= transition.keys()

    for payoff in manifest["payoffs"]:
        assert {
            "strategy_id",
            "state_id",
            "cycle",
            "cost",
            "utility",
            "currency",
        } <= payoff.keys()

    assert manifest["xla"]["status"] == "eligible_with_constraints"
    assert "bounded transition matrices" in manifest["xla"]["rationale"]


def test_treeage_style_manifest_is_artifact_first_and_small() -> None:
    """Fixtures should remain deterministic and independent of proprietary tools."""
    manifest = _load_json(TREEAGE_MANIFEST)

    assert manifest["provenance"]["source"] == "synthetic_fixture"
    assert manifest["provenance"]["treeage_file_required"] is False
    assert manifest["provenance"]["generated_by"] == "innovate ecosystem fixture"
    assert len(manifest["strategies"]) == 2
    assert len(manifest["transitions"]) <= 12
    assert len(manifest["payoffs"]) <= 12


def test_des_manifest_defines_event_log_queue_metrics_and_run_metadata() -> None:
    """DES fixtures should expose event logs and queue metrics, not engine state."""
    manifest = _load_json(DES_MANIFEST)

    assert manifest["schema_version"] == "0.1.0"
    assert manifest["artifact_type"] == "operational_modeling.des"
    assert manifest["engine_dependency"] == "none"
    assert manifest["private_engine_state"] == "excluded"

    required_sections = {
        "run_metadata",
        "event_log",
        "queue_metrics",
        "pathway_states",
        "resources",
        "ordering_rules",
        "provenance",
        "xla",
    }
    assert required_sections <= manifest.keys()

    assert {"simulation_id", "scenario_id", "run_id", "seed"} <= manifest["run_metadata"].keys()

    event_columns = [column["name"] for column in manifest["event_log"]["columns"]]
    assert event_columns == [
        "simulation_id",
        "run_id",
        "entity_id",
        "event_index",
        "event_time",
        "event_type",
        "state_before",
        "state_after",
        "resource_id",
        "queue_time",
    ]

    metric_columns = [column["name"] for column in manifest["queue_metrics"]["columns"]]
    assert metric_columns == [
        "simulation_id",
        "run_id",
        "resource_id",
        "mean_queue_time",
        "p95_queue_time",
        "utilization",
        "completed_entities",
    ]

    assert manifest["ordering_rules"] == [
        "sort by simulation_id",
        "sort by run_id",
        "sort by entity_id",
        "sort by event_index",
        "break ties by event_time",
    ]
    assert manifest["xla"]["status"] == "rejected_for_classic_des"
    assert "dynamic event queue" in manifest["xla"]["rationale"]


def test_des_events_are_deterministically_ordered_and_reference_known_resources() -> None:
    """The sample event rows should be small, ordered, and internally consistent."""
    manifest = _load_json(DES_MANIFEST)
    rows = manifest["event_log"]["rows"]
    resources = {resource["resource_id"] for resource in manifest["resources"]}
    states = {state["state_id"] for state in manifest["pathway_states"]}

    assert 0 < len(rows) <= 12
    assert rows == sorted(
        rows,
        key=lambda row: (
            row["simulation_id"],
            row["run_id"],
            row["entity_id"],
            row["event_index"],
            row["event_time"],
        ),
    )

    for row in rows:
        assert row["resource_id"] in resources
        assert row["state_before"] in states
        assert row["state_after"] in states
        assert row["queue_time"] >= 0

    for metric in manifest["queue_metrics"]["rows"]:
        assert metric["resource_id"] in resources
        assert metric["mean_queue_time"] >= 0
        assert metric["p95_queue_time"] >= metric["mean_queue_time"]
        assert 0 <= metric["utilization"] <= 1


def test_ecosystem_docs_link_operational_modeling_fixture_contracts() -> None:
    """Ecosystem docs should link fixtures to promotion gates and scope limits."""
    specs_readme = Path("specs/ecosystem/README.md").read_text()
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()

    for text in (specs_readme, strategy):
        normalized = " ".join(text.split()).lower()

        assert "operational_modeling/treeage_style/manifest.json" in text
        assert "operational_modeling/des/manifest.json" in text
        assert "adapter promotion ladder" in normalized
        assert "runtime simulation engines out of the current `innovate` package" in normalized
        assert "xla eligibility or rejection" in normalized
