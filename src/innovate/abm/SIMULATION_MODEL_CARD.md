# Model Card — Kairos Simulation Adapter

| Field | Value |
|-------|--------|
| **model_key** | `kairos_simulation` |
| **model_name** | KairosSimulationAdapter |
| **family** | simulation |
| **stability** | experimental (contract-stable schema `1.0`) |
| **import_path** | `innovate.abm.kairos_adapter.KairosSimulationAdapter` |

## Summary

Deterministic DES/ABM simulation adapter aligned with Kairos contracts.
Uses a Python reference engine while Kairos bridge crates remain gated.

## Assumptions

- Fixed seeds and topology produce deterministic traces.
- Bridge crates are not promoted unless smoke evidence says so.
- Mesa/NDLib are optional and not required for base installs.

## Inputs

- `KairosSimulationRequest` (seed, agents, topology, horizon, steps, interventions)

## Outputs

- Scheduler events, ABM behavior updates, DES trajectory events
- Policy/network diffusion traces
- JSON and Arrow-oriented telemetry artifacts
- Dependency evidence (revision, smoke, bridge gates)

## Diagnostics

- `adapter.status()` backend and bridge promotion flags
- `diagnostics.fallback_reason` when bridges are gated

## Limitations

- Native FFI/UniFFI/Diplomat dispatch is not enabled until bridge promotion.
- Reference engine implements contract semantics; it is not a full Kairos ECS reimplementation.
- Legacy Mesa ABM models remain separate and require `legacy-abm`.

## Benchmark case IDs

- `tests/unit/test_kairos_adapter_contract.py`
