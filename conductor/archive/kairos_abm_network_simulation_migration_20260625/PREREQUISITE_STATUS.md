# Phase 0 — Kairos Dependency Inclusion Prerequisite

**Track:** `kairos_abm_network_simulation_migration_20260625`
**Date:** 2026-07-09
**Prerequisite track:** `kairos_dependency_inclusion_20260626`

## Decision

**Proceed with behavior-level adapter work.** Core Kairos inclusion is established
in the repository even though the dependency-inclusion track still has open
formalization tasks (release-readiness evidence packaging, final CI gate).

Those remaining inclusion tasks do **not** block defining the Python adapter
contract, deterministic simulation semantics, legacy fail-safes, or docs.

## Evidence checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Kairos source revision | Pass | `bindings/rust/Cargo.toml` pins `edithatogo/kairos` @ `fae901558f07b7b717a676adbafbe2cdc78dea1c` |
| Selected crate set | Pass | `kairo-ecs-{types,core,state,rng,des,abm,arrow}` as git deps |
| Build plumbing | Pass | `bindings/rust/Cargo.lock` records revision; `tests/test_kairos_build_plumbing.py` |
| Smoke scenarios | Pass | `bindings/rust/examples/kairos_{des,abm}_smoke.rs`; `tests/test_kairos_smoke.py` |
| Integration report | Pass | `conductor/tracks/kairos_dependency_inclusion_20260626/KAIROS_INTEGRATION_REPORT.md` |
| Mesa/NDLib not base deps | Pass | Removed from `[project].dependencies`; available via `legacy-abm` extra |
| Bridge crates promoted | **Not promoted** | `kairo-ecs-ffi`, `kairo-ecs-uniffi`, `kairo-ecs-diplomat` remain gated; adapter must fail closed for unpromoted bridges |
| Inclusion track plan complete | Partial | Inclusion plan still has open release-evidence / CI formalization tasks |

## Blocker record

No hard blocker for adapter contract and deterministic Python-side adapter path.

**Soft dependency:** full "Kairos-backed simulation support" release claim remains
gated on inclusion-track release-readiness evidence and unpromoted bridge crates.
The adapter reports this honestly via dependency evidence and will not claim
promoted FFI/UniFFI/Diplomat bridges without smoke proof.

## Mesa / NDLib base-install status

- Base install does **not** require `mesa` or `ndlib` (`tests/test_kairos_dependency.py`).
- Optional install: `pip install innovate[legacy-abm]`.
- Legacy modules under `innovate.abm` still import Mesa/NDLib at module load;
  this migration track adds fail-safe loaders and migration notes so base-install
  users get clear guidance instead of silent support claims.
