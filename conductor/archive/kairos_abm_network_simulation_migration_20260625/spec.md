# Kairos ABM and Network Simulation Migration

## Overview

Move agent-based and network simulation surfaces toward a Kairos-backed adapter
strategy instead of relying on legacy Mesa-style assumptions. Kairos is now an
actual inclusion target rather than an optional future note. This track depends
on `kairos_dependency_inclusion_20260626`, which must establish the Kairos
dependency source, manifest migration, build smoke evidence, and removal of
Mesa/NDLib from the base install before this track implements behavior-level
adapter semantics.

The track should keep simulation artifacts reproducible and align
network/policy diffusion payloads with the kernel contract.

## Functional Requirements

- Audit current ABM, NDLib, network diffusion, and simulation adapter surfaces
  after the Kairos dependency inclusion track has established repo-first Kairos
  build evidence.
- Define a Kairos adapter contract for deterministic scheduler events,
  ECS-style agent state, DES trajectory/resource queue events, ABM behavior
  updates, deterministic random streams, Arrow/JSON telemetry artifacts, policy
  interventions, network topology, and simulation traces.
- Add compatibility shims or migration notes for existing ABM examples.
- Ensure simulation outputs are stable JSON/Arrow-compatible artifacts.
- Add tests for installed Kairos paths, dependency evidence freshness, and
  fail-safe behavior when an unpromoted bridge crate is unavailable.
- Update Starlight docs and model cards for the simulation adapter boundary.

## Non-Functional Requirements

- Kairos is the intended DES/ABM replacement direction and must not be described
  as optional future work after the dependency inclusion track completes.
- Simulations must be reproducible when seeds and inputs are fixed.
- Legacy adapter behavior must not silently claim Mesa/NDLib or Kairos support
  without proof.
- Bridge crates such as `kairo-ecs-ffi`, `kairo-ecs-uniffi`, and
  `kairo-ecs-diplomat` must remain gated by smoke evidence until promoted.

## Acceptance Criteria

- `kairos_dependency_inclusion_20260626` is complete or this track records a
  clear blocker before attempting behavior-level Kairos adapter work.
- Kairos adapter contract and tests exist.
- ABM/network simulation docs explain Kairos inclusion and legacy migration
  policy.
- Policy/network diffusion traces can be exported as stable artifacts.
- Release-readiness evidence records simulation adapter status.

## Out Of Scope

- Building a separate simulation platform.
- Repeating dependency inclusion and manifest migration already owned by
  `kairos_dependency_inclusion_20260626`.
