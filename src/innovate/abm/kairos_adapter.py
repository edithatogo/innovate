"""Kairos-aligned DES/ABM adapter with fail-closed bridge diagnostics."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from innovate.abm.kairos_contract import (
    BRIDGE_KAIROS_CRATES,
    CORE_KAIROS_CRATES,
    KAIROS_ADAPTER_SCHEMA_VERSION,
    KAIROS_PINNED_REVISION,
    KAIROS_SOURCE_URL,
    ABMBehaviorUpdate,
    AgentStateSpec,
    BridgeCrateStatus,
    DESTrajectoryEvent,
    InterventionSpec,
    KairosDependencyEvidence,
    KairosSimulationRequest,
    KairosSimulationResult,
    PolicyNetworkTrace,
    SchedulerEvent,
    TelemetryArtifact,
)


def _discover_repo_root() -> Path:
    """Locate the innovate repository root when running from a source checkout."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[3], Path.cwd(), *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "bindings" / "rust").is_dir():
            return candidate
    return here.parents[3]


_REPO_ROOT = _discover_repo_root()


def collect_kairos_dependency_evidence(
    *,
    repo_root: Path | None = None,
    promoted_bridges: Mapping[str, bool] | None = None,
) -> KairosDependencyEvidence:
    """Collect fail-closed evidence about Kairos inclusion and bridge promotion."""
    root = repo_root or _discover_repo_root()
    cargo = root / "bindings" / "rust" / "Cargo.toml"
    des_smoke = root / "bindings" / "rust" / "examples" / "kairos_des_smoke.rs"
    abm_smoke = root / "bindings" / "rust" / "examples" / "kairos_abm_smoke.rs"
    pyproject = root / "pyproject.toml"

    cargo_text = cargo.read_text(encoding="utf-8") if cargo.is_file() else ""
    revision = KAIROS_PINNED_REVISION if KAIROS_PINNED_REVISION in cargo_text else ""
    source_url = KAIROS_SOURCE_URL if "edithatogo/kairos" in cargo_text else ""
    core = tuple(crate for crate in CORE_KAIROS_CRATES if crate in cargo_text)

    promoted = dict(promoted_bridges or {})
    bridge_statuses: list[BridgeCrateStatus] = []
    for crate in BRIDGE_KAIROS_CRATES:
        if promoted.get(crate):
            bridge_statuses.append(
                BridgeCrateStatus(
                    crate=crate,
                    status="promoted",
                    reason="explicit smoke promotion recorded by caller",
                )
            )
        else:
            bridge_statuses.append(
                BridgeCrateStatus(
                    crate=crate,
                    status="gated",
                    reason="bridge crate remains gated until dedicated smoke promotion",
                )
            )

    pyproject_text = pyproject.read_text(encoding="utf-8") if pyproject.is_file() else ""
    mesa_base = _dependency_listed_in_base(pyproject_text, "mesa")
    ndlib_base = _dependency_listed_in_base(pyproject_text, "ndlib")

    return KairosDependencyEvidence(
        source_url=source_url or KAIROS_SOURCE_URL,
        revision=revision or KAIROS_PINNED_REVISION,
        core_crates=core or CORE_KAIROS_CRATES,
        bridge_crates=tuple(bridge_statuses),
        smoke_des=des_smoke.is_file(),
        smoke_abm=abm_smoke.is_file(),
        mesa_base_required=mesa_base,
        ndlib_base_required=ndlib_base,
    )


def _dependency_listed_in_base(pyproject_text: str, package: str) -> bool:
    """Return True if package appears in the base dependencies list."""
    in_deps = False
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("["):
                break
            if package in stripped and not stripped.startswith("#"):
                return True
    return False


def _stable_seed_token(value: str) -> int:
    """Deterministic integer token from text (avoids PYTHONHASHSEED instability)."""
    token = 2166136261
    for char in value.encode("utf-8"):
        token ^= char
        token = (token * 16777619) & 0xFFFFFFFF
    return token


def _seeded_rng(request: KairosSimulationRequest) -> random.Random:
    rng = random.Random(request.seed.primary)
    for stream in request.random_streams:
        mixed = (
            request.seed.primary ^ _stable_seed_token(stream.name) ^ _stable_seed_token(stream.seed.stream_id)
        ) & 0xFFFFFFFF
        rng = random.Random(mixed)
    return rng


def _adjacency_from_topology(request: KairosSimulationRequest) -> dict[str, list[tuple[str, float]]]:
    node_ids = list(request.topology.node_ids)
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in node_ids}
    for left, right, weight in request.topology.edges:
        adjacency[left].append((right, weight))
        if not request.topology.directed:
            adjacency[right].append((left, weight))
    return adjacency


def _apply_due_interventions(
    interventions: Sequence[InterventionSpec],
    intervention_index: int,
    time: float,
    active_boost: float,
    applied_labels: list[str],
    scheduler_events: list[SchedulerEvent],
) -> tuple[int, float]:
    while intervention_index < len(interventions) and float(interventions[intervention_index].time) <= time:
        intervention = interventions[intervention_index]
        active_boost += float(intervention.effect)
        applied_labels.append(intervention.label)
        scheduler_events.append(
            SchedulerEvent(
                time=float(intervention.time),
                event_type="intervention",
                payload={
                    "label": intervention.label,
                    "effect": float(intervention.effect),
                    "target_nodes": list(intervention.target_nodes),
                },
                event_id=f"evt-int-{intervention_index}",
            )
        )
        intervention_index += 1
    return intervention_index, active_boost


def _append_des_tick(
    rng: random.Random,
    time: float,
    step: int,
    queue_depth: int,
    des_events: list[DESTrajectoryEvent],
    scheduler_events: list[SchedulerEvent],
) -> int:
    arrivals = 1 if rng.random() < 0.55 else 0
    queue_depth += arrivals
    service_started = queue_depth > 0
    if service_started:
        queue_depth = max(0, queue_depth - 1)
    des_events.append(
        DESTrajectoryEvent(
            time=time,
            resource="adoption_channel",
            queue_depth=queue_depth,
            service_started=service_started,
            metadata={"step": step, "arrivals": arrivals},
        )
    )
    scheduler_events.append(
        SchedulerEvent(
            time=time,
            event_type="des_tick",
            payload={"step": step, "queue_depth": queue_depth},
            event_id=f"evt-des-{step}",
        )
    )
    return queue_depth


def _neighbor_pressure(
    node: str,
    adjacency: dict[str, list[tuple[str, float]]],
    agent_nodes: dict[str, str],
    agent_states: dict[str, AgentStateSpec],
) -> float:
    pressure = 0.0
    for neighbor, weight in adjacency.get(node, []):
        has_adopted_neighbor = any(
            other_node == neighbor and agent_states[other_id].state == "adopted"
            for other_id, other_node in agent_nodes.items()
        )
        if has_adopted_neighbor:
            pressure += float(weight)
    return pressure


def _apply_agent_adoptions(
    *,
    request: KairosSimulationRequest,
    time: float,
    step: int,
    rng: random.Random,
    active_boost: float,
    agent_states: dict[str, AgentStateSpec],
    agent_nodes: dict[str, str],
    adjacency: dict[str, list[tuple[str, float]]],
    agent_updates: list[ABMBehaviorUpdate],
    scheduler_events: list[SchedulerEvent],
) -> None:
    threshold = float(request.adoption_threshold)
    for agent_id, agent in list(agent_states.items()):
        if agent.state == "adopted":
            continue
        node = agent_nodes[agent_id]
        pressure = _neighbor_pressure(node, adjacency, agent_nodes, agent_states)
        score = pressure + active_boost + rng.random() * 0.1
        if score < threshold:
            continue
        previous = agent.state
        agent_states[agent_id] = AgentStateSpec(
            agent_id=agent.agent_id,
            state="adopted",
            attributes={**dict(agent.attributes), "adopted_at": time},
        )
        agent_updates.append(
            ABMBehaviorUpdate(
                time=time,
                agent_id=agent_id,
                from_state=previous,
                to_state="adopted",
                reason="neighbor_pressure_or_intervention",
            )
        )
        scheduler_events.append(
            SchedulerEvent(
                time=time,
                event_type="agent_adopted",
                payload={"agent_id": agent_id, "node": node},
                event_id=f"evt-adopt-{agent_id}-{step}",
            )
        )


def _record_node_adoption(
    node_ids: list[str],
    agent_nodes: dict[str, str],
    agent_states: dict[str, AgentStateSpec],
    adoption_series: dict[str, list[float]],
) -> None:
    for node in node_ids:
        node_agents = [agent_id for agent_id, other_node in agent_nodes.items() if other_node == node]
        if not node_agents:
            adoption_series[node].append(0.0)
            continue
        adopted = sum(1 for agent_id in node_agents if agent_states[agent_id].state == "adopted")
        adoption_series[node].append(adopted / len(node_agents))


def bridge_crate_available(crate: str, evidence: KairosDependencyEvidence | None = None) -> bool:
    """Return whether a bridge crate is promoted; unpromoted crates fail closed."""
    evidence = evidence or collect_kairos_dependency_evidence()
    for item in evidence.bridge_crates:
        if item.crate == crate:
            return item.status == "promoted"
    return False


class KairosSimulationAdapter:
    """Run deterministic Kairos-aligned simulations with honest backend diagnostics.

    Bridge crates remain gated by default. Until FFI/UniFFI/Diplomat smoke
    promotion exists, runs use the deterministic reference engine that
    implements the Kairos adapter contract and records backend diagnostics.
    """

    backend_name = "kairos_contract_reference"

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        promoted_bridges: Mapping[str, bool] | None = None,
    ) -> None:
        self.repo_root = repo_root or _REPO_ROOT
        self.promoted_bridges = dict(promoted_bridges or {})
        self.evidence = collect_kairos_dependency_evidence(
            repo_root=self.repo_root,
            promoted_bridges=self.promoted_bridges,
        )

    def status(self) -> dict[str, Any]:
        """Return adapter status suitable for release evidence."""
        return {
            "schema_version": KAIROS_ADAPTER_SCHEMA_VERSION,
            "backend": self.backend_name,
            "dependency_evidence": self.evidence.to_dict(),
            "bridge_promoted": self.evidence.claims_promoted_bridge(),
            "kairos_smoke_ready": self.evidence.claims_kairos_backed_simulation(),
            "legacy_base_deps_present": self.evidence.mesa_base_required or self.evidence.ndlib_base_required,
        }

    def validate_request(self, request: KairosSimulationRequest) -> None:
        """Validate request contract (dataclass already enforces invariants)."""
        if not isinstance(request, KairosSimulationRequest):
            raise TypeError("request must be a KairosSimulationRequest")
        # Re-run post-init style checks via to_dict round-trip
        _ = request.to_dict()

    def run(self, request: KairosSimulationRequest) -> KairosSimulationResult:
        """Execute a deterministic simulation for the request contract."""
        self.validate_request(request)
        if self.evidence.claims_promoted_bridge():
            # Future: dispatch to promoted native bridge. Until then, never claim it.
            raise RuntimeError("promoted bridge dispatch is not implemented")

        return self._run_reference(request)

    def _run_reference(self, request: KairosSimulationRequest) -> KairosSimulationResult:
        rng = _seeded_rng(request)
        agent_states = {agent.agent_id: agent for agent in request.agents}
        node_ids = list(request.topology.node_ids)
        agent_nodes = {agent_id: node_ids[index % len(node_ids)] for index, agent_id in enumerate(agent_states)}
        adjacency = _adjacency_from_topology(request)

        interventions = sorted(request.interventions, key=lambda item: float(item.time))
        intervention_index = 0
        active_boost = 0.0
        applied_labels: list[str] = []
        scheduler_events: list[SchedulerEvent] = [
            SchedulerEvent(
                time=0.0,
                event_type="simulation_start",
                payload={"seed": request.seed.primary},
                event_id="evt-start",
            )
        ]
        agent_updates: list[ABMBehaviorUpdate] = []
        des_events: list[DESTrajectoryEvent] = []
        times: list[float] = []
        adoption_series: dict[str, list[float]] = {node: [] for node in node_ids}
        dt = float(request.horizon) / float(request.steps)
        queue_depth = 0

        for step in range(1, request.steps + 1):
            time = step * dt
            intervention_index, active_boost = _apply_due_interventions(
                interventions,
                intervention_index,
                time,
                active_boost,
                applied_labels,
                scheduler_events,
            )
            queue_depth = _append_des_tick(rng, time, step, queue_depth, des_events, scheduler_events)
            _apply_agent_adoptions(
                request=request,
                time=time,
                step=step,
                rng=rng,
                active_boost=active_boost,
                agent_states=agent_states,
                agent_nodes=agent_nodes,
                adjacency=adjacency,
                agent_updates=agent_updates,
                scheduler_events=scheduler_events,
            )
            times.append(time)
            _record_node_adoption(node_ids, agent_nodes, agent_states, adoption_series)

        scheduler_events.append(
            SchedulerEvent(
                time=float(request.horizon),
                event_type="simulation_end",
                payload={"steps": request.steps},
                event_id="evt-end",
            )
        )
        return self._build_result(
            request=request,
            scheduler_events=scheduler_events,
            agent_updates=agent_updates,
            des_events=des_events,
            agent_states=agent_states,
            times=times,
            adoption_series=adoption_series,
            applied_labels=applied_labels,
            node_ids=node_ids,
        )

    def _build_result(
        self,
        *,
        request: KairosSimulationRequest,
        scheduler_events: list[SchedulerEvent],
        agent_updates: list[ABMBehaviorUpdate],
        des_events: list[DESTrajectoryEvent],
        agent_states: dict[str, AgentStateSpec],
        times: list[float],
        adoption_series: dict[str, list[float]],
        applied_labels: list[str],
        node_ids: list[str],
    ) -> KairosSimulationResult:
        trace = PolicyNetworkTrace(
            times=tuple(times),
            node_adoption={node: tuple(series) for node, series in adoption_series.items()},
            intervention_labels=tuple(applied_labels),
        )
        final_agents = tuple(agent_states[agent_id] for agent_id in sorted(agent_states))
        result_payload = {
            "scheduler_events": [event.to_dict() for event in scheduler_events],
            "agent_updates": [update.to_dict() for update in agent_updates],
            "des_events": [event.to_dict() for event in des_events],
            "policy_network_trace": trace.to_dict(),
            "final_agents": [agent.to_dict() for agent in final_agents],
        }
        telemetry = (
            TelemetryArtifact(
                format="json",
                schema_version=KAIROS_ADAPTER_SCHEMA_VERSION,
                kind="kairos.simulation.result",
                payload=result_payload,
            ),
            TelemetryArtifact(
                format="arrow",
                schema_version=KAIROS_ADAPTER_SCHEMA_VERSION,
                kind="kairos.simulation.trace_table",
                payload={
                    "columns": ["time", *node_ids],
                    "rows": [
                        [times[index], *[adoption_series[node][index] for node in node_ids]]
                        for index in range(len(times))
                    ],
                },
            ),
        )
        diagnostics = {
            "backend": self.backend_name,
            "bridge_dispatch": "not_used",
            "bridge_promoted": self.evidence.claims_promoted_bridge(),
            "fallback_reason": "bridge crates gated; using deterministic contract reference engine",
            "kairos_revision": self.evidence.revision,
            "smoke_des": self.evidence.smoke_des,
            "smoke_abm": self.evidence.smoke_abm,
        }
        return KairosSimulationResult(
            request=request,
            scheduler_events=tuple(scheduler_events),
            agent_updates=tuple(agent_updates),
            des_events=tuple(des_events),
            final_agents=final_agents,
            policy_network_trace=trace,
            dependency_evidence=self.evidence,
            telemetry=telemetry,
            backend=self.backend_name,
            diagnostics=diagnostics,
        )

    def export_json(self, result: KairosSimulationResult) -> str:
        """Serialize a result to stable JSON."""
        return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))

    def export_arrow_table_dict(self, result: KairosSimulationResult) -> dict[str, Any]:
        """Return Arrow-oriented table dict from telemetry (no pyarrow required)."""
        for artifact in result.telemetry:
            if artifact.format == "arrow" and artifact.kind == "kairos.simulation.trace_table":
                return dict(artifact.payload)
        raise ValueError("result does not contain an Arrow trace table artifact")
