# Kairos Dependency Inclusion

## Overview

Include Kairos as the actual DES/ABM migration dependency source for Innovate.
This track owns dependency inclusion, manifest migration, build plumbing, and
release-evidence truthfulness only. It must start from the Kairos repository
workspace at `https://github.com/edithatogo/kairos`, because that repository
currently exposes the relevant `kairo-ecs-*` Rust crates directly.

This track replaces the previous "optional future adapter" posture with an
actual inclusion baseline. The follow-on Kairos ABM and Network Simulation
Migration track owns behavior-level adapter implementation after this track
establishes the dependency and build foundation.

## Dependency Sources

The implementation must use the Kairos repository as the first integration
source for these crates:

- `kairo-ecs-types`
- `kairo-ecs-core`
- `kairo-ecs-state`
- `kairo-ecs-rng`
- `kairo-ecs-des`
- `kairo-ecs-abm`
- `kairo-ecs-arrow`

The implementation may use matching crates.io packages only if the matching
`kairo-ecs-*` crates are published, versioned, and proven compatible with the
target Kairos repository state.

Bridge crates are candidates, not assumed stable dependencies:

- `kairo-ecs-ffi`
- `kairo-ecs-uniffi`
- `kairo-ecs-diplomat`

Bridge crates must be gated behind build and smoke evidence because Kairos
marks bridge and facade surfaces as in review.

## Functional Requirements

- Add Kairos dependency plumbing for Innovate's Rust/toolchain lane using the
  Kairos repository workspace as the primary source.
- Remove `mesa` and `ndlib` from base Python runtime dependencies.
- Reclassify `networkx` separately:
  - keep it only if it remains needed for plotting or graph utility APIs;
  - otherwise move it behind a graph or visualization extra.
- Add build or smoke evidence proving the selected Kairos crates compile in
  Innovate's Rust/toolchain lane.
- Add a minimal Kairos DES smoke scenario that proves deterministic event
  scheduling can be built and invoked through the selected integration path.
- Add a minimal Kairos ABM smoke scenario that proves ECS-style agent state and
  behavior update plumbing can be built and invoked through the selected
  integration path.
- Record external compatibility constraints if package registries, Python
  packaging, Rust packaging, or scientific dependencies block a Python
  3.14-only Kairos-backed baseline.
- Update release-readiness evidence so the project states Kairos inclusion
  status truthfully.

## Non-Functional Requirements

- The base install must not require Mesa or NDLib after this track is complete.
- Kairos inclusion must be explicit and testable rather than implied by docs.
- The dependency policy must preserve reproducibility by pinning or recording
  the Kairos repository revision used for smoke evidence.
- Bridge crate adoption must fail closed when the bridge smoke tests fail or
  when the Kairos repository marks those surfaces as unstable.
- The track must not silently lower the Python 3.14 baseline to satisfy a
  dependency without recording the blocker as an external compatibility
  constraint.

## Acceptance Criteria

- Innovate's Rust/toolchain lane can build the selected Kairos crate set.
- A minimal Kairos DES smoke scenario passes in the target toolchain.
- A minimal Kairos ABM smoke scenario passes in the target toolchain.
- `mesa` and `ndlib` are no longer base runtime requirements.
- `networkx` is either retained with a specific plotting/graph utility reason
  or moved behind a graph/visualization extra.
- Release evidence includes the Kairos repository source, selected revision or
  published crate versions, build status, smoke status, bridge-crate status,
  and any registry compatibility constraints.
- The follow-on Kairos ABM and Network Simulation Migration track can rely on
  this track as the dependency inclusion prerequisite.

## Out Of Scope

- Implementing the full Innovate simulation adapter behavior.
- Migrating every ABM or network diffusion API to Kairos semantics.
- Publishing Kairos crates or changing the Kairos upstream repository.
- Treating bridge crates as stable without smoke evidence.
