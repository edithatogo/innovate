"""TDD coverage for Kairos ABM/DES adapter contracts and deterministic runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from innovate.abm.kairos_adapter import (
    KairosSimulationAdapter,
    bridge_crate_available,
    collect_kairos_dependency_evidence,
)
from innovate.abm.kairos_contract import (
    BRIDGE_KAIROS_CRATES,
    KAIROS_PINNED_REVISION,
    AgentStateSpec,
    InterventionSpec,
    KairosSimulationRequest,
    RandomStreamConfig,
    SimulationSeed,
    TopologySpec,
)
from innovate.abm.legacy import (
    LEGACY_ABM_EXTRA,
    LegacyABMDependencyError,
    load_legacy_module,
    migration_guidance,
    require_legacy_stack,
)


def _sample_request(*, seed: int = 42) -> KairosSimulationRequest:
    topology = TopologySpec.from_edge_list(
        node_ids=["n0", "n1", "n2"],
        edges=[("n0", "n1", 1.0), ("n1", "n2", 1.0)],
    )
    agents = (
        AgentStateSpec(agent_id="a0", state="susceptible", attributes={"influence": 0.2}),
        AgentStateSpec(agent_id="a1", state="adopted", attributes={"influence": 0.8}),
        AgentStateSpec(agent_id="a2", state="susceptible", attributes={"influence": 0.1}),
    )
    return KairosSimulationRequest(
        seed=SimulationSeed(primary=seed, stream_id="main"),
        agents=agents,
        topology=topology,
        horizon=10.0,
        steps=10,
        interventions=(
            InterventionSpec(time=3.0, label="subsidy", effect=0.4, target_nodes=("n2",)),
        ),
        random_streams=(
            RandomStreamConfig(name="behavior", seed=SimulationSeed(primary=seed, stream_id="behavior")),
        ),
        adoption_threshold=0.35,
    )


def test_topology_rejects_unknown_nodes() -> None:
    with pytest.raises(ValueError, match="edge endpoints"):
        TopologySpec(node_ids=("a",), edges=(("a", "missing", 1.0),))


def test_request_requires_positive_horizon_and_unique_agents() -> None:
    topology = TopologySpec.from_edge_list(["n0"], [])
    with pytest.raises(ValueError, match="horizon"):
        KairosSimulationRequest(
            seed=SimulationSeed(primary=1),
            agents=(AgentStateSpec(agent_id="a0", state="susceptible"),),
            topology=topology,
            horizon=0.0,
            steps=1,
        )
    with pytest.raises(ValueError, match="unique"):
        KairosSimulationRequest(
            seed=SimulationSeed(primary=1),
            agents=(
                AgentStateSpec(agent_id="a0", state="susceptible"),
                AgentStateSpec(agent_id="a0", state="adopted"),
            ),
            topology=topology,
            horizon=1.0,
            steps=1,
        )


def test_dependency_evidence_is_fail_closed_for_bridges(tmp_path: Path) -> None:
    evidence = collect_kairos_dependency_evidence()
    assert evidence.revision == KAIROS_PINNED_REVISION
    assert evidence.smoke_des is True
    assert evidence.smoke_abm is True
    assert evidence.mesa_base_required is False
    assert evidence.ndlib_base_required is False
    assert evidence.claims_promoted_bridge() is False
    assert evidence.claims_kairos_backed_simulation() is True
    for crate in BRIDGE_KAIROS_CRATES:
        assert bridge_crate_available(crate, evidence) is False


def test_adapter_status_reports_honest_backend() -> None:
    adapter = KairosSimulationAdapter()
    status = adapter.status()
    assert status["backend"] == "kairos_contract_reference"
    assert status["bridge_promoted"] is False
    assert status["kairos_smoke_ready"] is True
    assert status["legacy_base_deps_present"] is False


def test_deterministic_scheduler_and_streams() -> None:
    adapter = KairosSimulationAdapter()
    request = _sample_request(seed=7)
    first = adapter.run(request)
    second = adapter.run(request)
    assert first.to_dict() == second.to_dict()
    assert first.scheduler_events
    assert first.scheduler_events[0].event_type == "simulation_start"
    assert first.scheduler_events[-1].event_type == "simulation_end"
    assert any(event.event_type == "intervention" for event in first.scheduler_events)
    assert first.des_events
    assert all(event.resource == "adoption_channel" for event in first.des_events)


def test_ecs_style_agent_updates_and_final_state() -> None:
    adapter = KairosSimulationAdapter()
    result = adapter.run(_sample_request(seed=11))
    assert result.final_agents
    assert {agent.agent_id for agent in result.final_agents} == {"a0", "a1", "a2"}
    # Seeded run should produce at least the initially adopted agent.
    assert any(agent.state == "adopted" for agent in result.final_agents)
    for update in result.agent_updates:
        assert update.from_state
        assert update.to_state == "adopted"


def test_policy_network_trace_and_telemetry_artifacts() -> None:
    adapter = KairosSimulationAdapter()
    result = adapter.run(_sample_request(seed=3))
    trace = result.policy_network_trace
    assert len(trace.times) == 10
    assert set(trace.node_adoption) == {"n0", "n1", "n2"}
    assert "subsidy" in trace.intervention_labels
    formats = {artifact.format for artifact in result.telemetry}
    assert formats == {"json", "arrow"}
    payload = json.loads(adapter.export_json(result))
    assert payload["schema_version"] == "1.0"
    assert payload["dependency_evidence"]["claims_promoted_bridge"] is False
    arrow_table = adapter.export_arrow_table_dict(result)
    assert "columns" in arrow_table and "rows" in arrow_table
    assert len(arrow_table["rows"]) == 10


def test_different_seeds_change_outcomes() -> None:
    adapter = KairosSimulationAdapter()
    left = adapter.run(_sample_request(seed=1))
    right = adapter.run(_sample_request(seed=999))
    assert left.to_dict() != right.to_dict()


def test_legacy_migration_guidance_and_fail_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    guidance = migration_guidance()
    assert guidance["legacy_extra"] == LEGACY_ABM_EXTRA
    assert "kairos_adapter" in guidance["replacement"]

    import innovate.abm.legacy as legacy_mod

    def _boom(name: str) -> None:
        raise ImportError(f"No module named '{name.split('.')[-1]}'")

    monkeypatch.setattr(legacy_mod, "import_module", _boom)
    with pytest.raises(LegacyABMDependencyError, match="legacy-abm"):
        load_legacy_module("model")
    with pytest.raises(LegacyABMDependencyError):
        require_legacy_stack()
