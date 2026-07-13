# Simulation Surface Audit

**Track:** `kairos_abm_network_simulation_migration_20260625`
**Date:** 2026-07-09
**Prerequisite:** See `PREREQUISITE_STATUS.md`

## Inventory

### Python ABM (`src/innovate/abm/`)

| Module | Backend | Base-install safe? | Notes |
|--------|---------|--------------------|-------|
| `agent.py` | Mesa `Agent` | No | Hard import of Mesa |
| `model.py` | Mesa `Model` / `MultiGrid` | No | Core innovation ABM |
| `competitive_diffusion.py` | Mesa | No | Multi-innovation competition |
| `disruptive_innovation.py` | Mesa | No | Disruptive performance dynamics |
| `sentiment_hype_cycle.py` | Mesa | No | Sentiment-driven adoption |
| `ndlib_model.py` | Mesa + NDLib | No | Network epidemic-style diffusion |

### Network / policy diffusion (kernel-aligned, not Mesa)

| Module | Role |
|--------|------|
| `models/network.py` | NetworkDiffusionModel with adjacency spillover |
| `models/policy.py` | Policy-aware diffusion |
| `models/contracts.py` | `NetworkDiffusionInputs`, `PolicyTimingInputs` |
| `policy/intervention.py` | Policy intervention helpers |

### Examples & tutorials

| Surface | Status |
|---------|--------|
| `examples/abm_examples.py` | Legacy Mesa-oriented |
| `examples/network_diffusion_example.py` | Network model path |
| `docs/.../tutorials/ndlib-integration.md` | Documents NDLib (legacy extra) |
| Kairos tutorial | **Missing** — add this track |

### Dependency evidence

| Surface | Status |
|---------|--------|
| `pyproject.toml` base deps | No mesa/ndlib |
| `legacy-abm` extra | mesa + ndlib optional |
| Rust Kairos git deps | Pinned revision in `bindings/rust/Cargo.toml` |
| Smoke examples | `kairos_des_smoke.rs`, `kairos_abm_smoke.rs` |
| Bridge crates | Not promoted |

## Gaps this track closes

1. No Python Kairos adapter contract or deterministic run path.
2. Legacy Mesa modules fail opaquely without `legacy-abm` extra.
3. No JSON/Arrow simulation telemetry contract for ABM/DES.
4. Docs still present NDLib as primary without Kairos policy.
5. Release evidence does not describe simulation adapter status.

## Target architecture

```
innovate.abm.kairos_contract  -> validated input/output contracts
innovate.abm.kairos_adapter   -> deterministic adapter + dependency evidence
innovate.abm.legacy           -> fail-safe loaders / migration diagnostics
```

Bridge crates remain unpromoted; the adapter uses a deterministic reference
engine that implements the Kairos-aligned contract and reports bridge status
honestly until FFI/UniFFI/Diplomat smoke promotion lands.
