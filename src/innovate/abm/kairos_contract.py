"""Validated contracts for Kairos-aligned DES/ABM simulation adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

KAIROS_ADAPTER_SCHEMA_VERSION = "1.0"
KAIROS_SOURCE_URL = "https://github.com/edithatogo/kairos.git"
KAIROS_PINNED_REVISION = "fae901558f07b7b717a676adbafbe2cdc78dea1c"
CORE_KAIROS_CRATES: tuple[str, ...] = (
    "kairo-ecs-types",
    "kairo-ecs-core",
    "kairo-ecs-state",
    "kairo-ecs-rng",
    "kairo-ecs-des",
    "kairo-ecs-abm",
    "kairo-ecs-arrow",
)
BRIDGE_KAIROS_CRATES: tuple[str, ...] = (
    "kairo-ecs-ffi",
    "kairo-ecs-uniffi",
    "kairo-ecs-diplomat",
)

BridgePromotionStatus = Literal["promoted", "gated", "unavailable"]
TelemetryFormat = Literal["json", "arrow"]


def _require_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SimulationSeed:
    """Primary and stream-split seeds for deterministic random streams."""

    primary: int
    stream_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.primary, int) or isinstance(self.primary, bool):
            raise ValueError("primary seed must be an integer")
        _require_non_empty_str(self.stream_id, "stream_id")

    def to_dict(self) -> dict[str, object]:
        return {"primary": self.primary, "stream_id": self.stream_id}


@dataclass(frozen=True, slots=True)
class RandomStreamConfig:
    """Configuration for a named deterministic random stream."""

    name: str
    seed: SimulationSeed
    algorithm: str = "pcg64"

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, "name")
        _require_non_empty_str(self.algorithm, "algorithm")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed.to_dict(),
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True, slots=True)
class TopologySpec:
    """Network topology used by network/policy diffusion traces."""

    node_ids: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]
    directed: bool = False

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ValueError("node_ids must be non-empty")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        node_set = set(self.node_ids)
        for left, right, weight in self.edges:
            if left not in node_set or right not in node_set:
                raise ValueError(f"edge endpoints must be known nodes: {(left, right)}")
            if float(weight) < 0.0:
                raise ValueError("edge weights must be non-negative")

    @classmethod
    def from_edge_list(
        cls,
        node_ids: Sequence[str],
        edges: Sequence[tuple[str, str] | tuple[str, str, float]],
        *,
        directed: bool = False,
    ) -> TopologySpec:
        normalized: list[tuple[str, str, float]] = []
        for edge in edges:
            if len(edge) == 2:
                left, right = edge
                weight = 1.0
            elif len(edge) == 3:
                left, right, weight = edge  # type: ignore[misc]
            else:
                raise ValueError("edges must be (source, target) or (source, target, weight)")
            normalized.append((str(left), str(right), float(weight)))
        return cls(node_ids=tuple(str(n) for n in node_ids), edges=tuple(normalized), directed=directed)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "edges": [[left, right, weight] for left, right, weight in self.edges],
            "directed": self.directed,
        }


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """Timed policy intervention applied during a simulation run."""

    time: float
    label: str
    effect: float
    target_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.time, (int, float)) or isinstance(self.time, bool):
            raise ValueError("time must be numeric")
        if float(self.time) < 0.0:
            raise ValueError("time must be non-negative")
        _require_non_empty_str(self.label, "label")

    def to_dict(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "label": self.label,
            "effect": float(self.effect),
            "target_nodes": list(self.target_nodes),
        }


@dataclass(frozen=True, slots=True)
class AgentStateSpec:
    """Initial ECS-style agent state."""

    agent_id: str
    state: str
    attributes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.agent_id, "agent_id")
        _require_non_empty_str(self.state, "state")
        object.__setattr__(self, "attributes", {str(k): float(v) for k, v in self.attributes.items()})

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """Deterministic scheduler event entry."""

    time: float
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self) -> None:
        if float(self.time) < 0.0:
            raise ValueError("event time must be non-negative")
        _require_non_empty_str(self.event_type, "event_type")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "event_id": self.event_id,
        }


@dataclass(frozen=True, slots=True)
class DESTrajectoryEvent:
    """DES trajectory or resource-queue event."""

    time: float
    resource: str
    queue_depth: int
    service_started: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.time) < 0.0:
            raise ValueError("time must be non-negative")
        _require_non_empty_str(self.resource, "resource")
        _require_non_negative_int(self.queue_depth, "queue_depth")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "resource": self.resource,
            "queue_depth": self.queue_depth,
            "service_started": bool(self.service_started),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ABMBehaviorUpdate:
    """Record of an agent behavior/state transition."""

    time: float
    agent_id: str
    from_state: str
    to_state: str
    reason: str = ""

    def __post_init__(self) -> None:
        if float(self.time) < 0.0:
            raise ValueError("time must be non-negative")
        for name in ("agent_id", "from_state", "to_state"):
            _require_non_empty_str(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "agent_id": self.agent_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BridgeCrateStatus:
    """Promotion status for a Kairos bridge crate."""

    crate: str
    status: BridgePromotionStatus
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.crate, "crate")
        if self.status not in ("promoted", "gated", "unavailable"):
            raise ValueError(f"invalid bridge status: {self.status}")
        _require_non_empty_str(self.reason, "reason")

    def to_dict(self) -> dict[str, object]:
        return {"crate": self.crate, "status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class KairosDependencyEvidence:
    """Fail-closed evidence about Kairos source and bridge promotion."""

    source_url: str
    revision: str
    core_crates: tuple[str, ...]
    bridge_crates: tuple[BridgeCrateStatus, ...]
    smoke_des: bool
    smoke_abm: bool
    mesa_base_required: bool
    ndlib_base_required: bool

    def claims_promoted_bridge(self) -> bool:
        return any(item.status == "promoted" for item in self.bridge_crates)

    def claims_kairos_backed_simulation(self) -> bool:
        """True only when core smoke evidence and no unpromoted-claim lie exist."""
        return bool(self.smoke_des and self.smoke_abm and self.revision)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_url": self.source_url,
            "revision": self.revision,
            "core_crates": list(self.core_crates),
            "bridge_crates": [item.to_dict() for item in self.bridge_crates],
            "smoke_des": self.smoke_des,
            "smoke_abm": self.smoke_abm,
            "mesa_base_required": self.mesa_base_required,
            "ndlib_base_required": self.ndlib_base_required,
            "claims_promoted_bridge": self.claims_promoted_bridge(),
            "claims_kairos_backed_simulation": self.claims_kairos_backed_simulation(),
        }


@dataclass(frozen=True, slots=True)
class KairosSimulationRequest:
    """Full request contract for a Kairos-aligned simulation run."""

    seed: SimulationSeed
    agents: tuple[AgentStateSpec, ...]
    topology: TopologySpec
    horizon: float
    steps: int
    interventions: tuple[InterventionSpec, ...] = ()
    random_streams: tuple[RandomStreamConfig, ...] = ()
    adoption_threshold: float = 0.5
    schema_version: str = KAIROS_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != KAIROS_ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if float(self.horizon) <= 0.0:
            raise ValueError("horizon must be positive")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not self.agents:
            raise ValueError("agents must be non-empty")
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_id values must be unique")
        if not 0.0 <= float(self.adoption_threshold) <= 1.0:
            raise ValueError("adoption_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed.to_dict(),
            "agents": [agent.to_dict() for agent in self.agents],
            "topology": self.topology.to_dict(),
            "horizon": float(self.horizon),
            "steps": self.steps,
            "interventions": [item.to_dict() for item in self.interventions],
            "random_streams": [item.to_dict() for item in self.random_streams],
            "adoption_threshold": float(self.adoption_threshold),
        }


@dataclass(frozen=True, slots=True)
class PolicyNetworkTrace:
    """Stable policy/network diffusion trace artifact."""

    times: tuple[float, ...]
    node_adoption: Mapping[str, tuple[float, ...]]
    intervention_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("times must be non-empty")
        n = len(self.times)
        for node, series in self.node_adoption.items():
            if len(series) != n:
                raise ValueError(f"adoption series for {node} must match times length")
        object.__setattr__(
            self,
            "node_adoption",
            {str(k): tuple(float(v) for v in series) for k, series in self.node_adoption.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "times": list(self.times),
            "node_adoption": {node: list(series) for node, series in self.node_adoption.items()},
            "intervention_labels": list(self.intervention_labels),
        }


@dataclass(frozen=True, slots=True)
class TelemetryArtifact:
    """JSON/Arrow-compatible simulation telemetry payload."""

    format: TelemetryFormat
    schema_version: str
    kind: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format not in ("json", "arrow"):
            raise ValueError(f"unsupported telemetry format: {self.format}")
        _require_non_empty_str(self.schema_version, "schema_version")
        _require_non_empty_str(self.kind, "kind")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class KairosSimulationResult:
    """Validated result of a Kairos-aligned simulation run."""

    request: KairosSimulationRequest
    scheduler_events: tuple[SchedulerEvent, ...]
    agent_updates: tuple[ABMBehaviorUpdate, ...]
    des_events: tuple[DESTrajectoryEvent, ...]
    final_agents: tuple[AgentStateSpec, ...]
    policy_network_trace: PolicyNetworkTrace
    dependency_evidence: KairosDependencyEvidence
    telemetry: tuple[TelemetryArtifact, ...]
    backend: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": KAIROS_ADAPTER_SCHEMA_VERSION,
            "backend": self.backend,
            "request": self.request.to_dict(),
            "scheduler_events": [event.to_dict() for event in self.scheduler_events],
            "agent_updates": [update.to_dict() for update in self.agent_updates],
            "des_events": [event.to_dict() for event in self.des_events],
            "final_agents": [agent.to_dict() for agent in self.final_agents],
            "policy_network_trace": self.policy_network_trace.to_dict(),
            "dependency_evidence": self.dependency_evidence.to_dict(),
            "telemetry": [item.to_dict() for item in self.telemetry],
            "diagnostics": dict(self.diagnostics),
        }
