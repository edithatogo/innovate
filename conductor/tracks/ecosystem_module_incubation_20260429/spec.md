# Ecosystem Module Incubation

## Overview

Define how `innovate` should participate in a health economics and outcomes
research (HEOR) ecosystem without taking ownership of health-economic
simulation, VOI analysis, generic surrogate modelling, or workflow
orchestration. `innovate` should remain the health-intervention diffusion,
adoption, implementation-spread, and policy-diffusion package.

## Ecosystem Roles

- `innovate` owns health-intervention adoption curves, diffusion dynamics,
  policy spread, network adoption, substitution, competition, and implementation
  uptake artifacts.
- `lifecourse` owns health-economic simulation, scenarios, run bundles, and
  reporting outputs.
- `voiage` owns Value of Information methods, VOI result objects, and VOI
  metamodel workflows.
- `mars` owns a fixed-API MARS surrogate/metamodel package. `innovate` may use
  it as an optional modelling backend only through public APIs.
- HEOML owns portable health-economic artifacts and extension namespaces.

## Goals

- Define `innovate` as the health-intervention adoption/diffusion sibling in
  the HEOR ecosystem.
- Define how health-intervention adoption and diffusion uncertainty can feed
  `lifecourse` scenarios and `voiage` VOI workflows.
- Keep integrations optional, artifact-first, and schema-versioned.
- Reserve a HEOML extension alignment for uptake, adoption, diffusion, and
  policy-spread artifacts.
- Avoid adding `lifecourse`, `voiage`, or `mars` to base runtime dependencies.

## Functional Requirements

- Document the ecosystem boundary for `innovate`, `lifecourse`, `voiage`,
  `mars`, HEOML, and future sibling modules.
- Require future sibling modules and integrations to have a clear HEOR role.
- Define candidate artifacts that `innovate` should produce:
  adoption curves, uptake trajectories, network diffusion traces, policy
  intervention metadata, parameter draws, uncertainty summaries, diagnostics,
  and model provenance.
- Define candidate artifacts that `innovate` may consume:
  scenario definitions, intervention metadata, population strata, calibration
  targets, and optional surrogate models.
- Define optional adapter policy for `lifecourse`, `voiage`, and `mars`.
- Add compatibility-fixture expectations for cross-repo validation.

## Non-Functional Requirements

- No sibling project dependency should enter the base `innovate` install through
  this planning track.
- Stable integrations must use public APIs, Arrow/Parquet/JSON artifacts, and
  versioned schemas rather than private Python internals.
- The existing `innovate` functional kernel and language-binding contracts must
  remain the stable portability surface.
- Pickle must not be part of the portable ecosystem contract.
- Generic non-HEOR innovation or diffusion modelling is out of scope for this
  ecosystem plan.

## Acceptance Criteria

- `docs/ecosystem/module_incubation_strategy.md` documents `innovate` ecosystem
  boundaries, candidate artifacts, HEOML alignment, optional integration policy,
  and promotion criteria.
- `specs/ecosystem/README.md` defines the first ecosystem contract outline.
- `conductor/tracks.md`, `documents/todo.md`, and `CHANGELOG.md` reference the
  ecosystem-module incubation work.

## Out Of Scope

- Implementing concrete adapters.
- Adding runtime dependencies on `lifecourse`, `voiage`, or `mars`.
- Creating new external repositories.
- Changing the `mars` core API.
- Replacing existing Arrow interchange or functional-kernel contracts.
