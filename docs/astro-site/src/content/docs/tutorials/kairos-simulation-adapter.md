---
title: Kairos Simulation Adapter
description: Deterministic DES/ABM simulation contracts, telemetry, and legacy Mesa/NDLib migration policy.
---

# Kairos Simulation Adapter

Innovate's supported agent-based and discrete-event simulation path is the
**Kairos-aligned adapter**. Mesa and NDLib are no longer base runtime
dependencies; they remain available only through the optional `legacy-abm`
extra.

## Policy

| Surface | Status |
|---------|--------|
| Kairos core crates (Rust git deps) | Included at a pinned revision |
| Bridge crates (`kairo-ecs-ffi`, `uniffi`, `diplomat`) | **Gated** until dedicated smoke promotion |
| Python adapter contract | Supported (`innovate.abm.kairos_contract`) |
| Deterministic reference engine | Supported (`KairosSimulationAdapter`) |
| Mesa / NDLib | Optional `legacy-abm` extra only |

The adapter never claims promoted bridge support without explicit promotion
evidence.

## Minimal example

```python
from innovate.abm import (
    AgentStateSpec,
    InterventionSpec,
    KairosSimulationAdapter,
    KairosSimulationRequest,
    SimulationSeed,
    TopologySpec,
)

request = KairosSimulationRequest(
    seed=SimulationSeed(primary=42),
    agents=(
        AgentStateSpec(agent_id="a0", state="susceptible"),
        AgentStateSpec(agent_id="a1", state="adopted"),
        AgentStateSpec(agent_id="a2", state="susceptible"),
    ),
    topology=TopologySpec.from_edge_list(
        ["n0", "n1", "n2"],
        [("n0", "n1", 1.0), ("n1", "n2", 1.0)],
    ),
    horizon=10.0,
    steps=10,
    interventions=(InterventionSpec(time=3.0, label="subsidy", effect=0.4),),
    adoption_threshold=0.35,
)

adapter = KairosSimulationAdapter()
result = adapter.run(request)

print(adapter.status()["backend"])
print(result.policy_network_trace.intervention_labels)
print(adapter.export_json(result)[:120], "...")
```

## Artifact schemas

Results use schema version `1.0` (`KAIROS_ADAPTER_SCHEMA_VERSION`).

### JSON telemetry (`kind: kairos.simulation.result`)

Includes:

- `scheduler_events` — deterministic event log
- `agent_updates` — ECS-style state transitions
- `des_events` — trajectory / resource-queue ticks
- `policy_network_trace` — per-node adoption series + intervention labels
- `dependency_evidence` — Kairos revision, smoke flags, bridge gate status

### Arrow-oriented table dict (`kind: kairos.simulation.trace_table`)

```json
{
  "columns": ["time", "n0", "n1", "n2"],
  "rows": [[1.0, 0.0, 1.0, 0.0]]
}
```

## Legacy Mesa / NDLib

```python
from innovate.abm.legacy import migration_guidance, load_legacy_module

print(migration_guidance()["install"])  # pip install innovate[legacy-abm]

# Fail-safe loader with clear migration diagnostics:
try:
    model_mod = load_legacy_module("model")
except Exception as exc:
    print(exc)
```

For network diffusion that does **not** need Mesa, prefer
`innovate.models.network.NetworkDiffusionModel` and the functional kernel.

## Dependency evidence

```python
from innovate.abm import collect_kairos_dependency_evidence

evidence = collect_kairos_dependency_evidence()
assert evidence.claims_promoted_bridge() is False
assert evidence.mesa_base_required is False
```

See also:

- Conductor track `kairos_abm_network_simulation_migration_20260625`
- Kairos inclusion report under `kairos_dependency_inclusion_20260626`
