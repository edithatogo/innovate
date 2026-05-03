# Specification: Lifecourse Adoption-Trajectory Fixture

## Overview

Define the first deterministic adoption-trajectory fixture that `lifecourse`
can consume for health-economic scenario workflows without importing
`innovate` internals. This track converts the ecosystem incubation follow-up
"Define a minimal adoption-trajectory fixture that `lifecourse` can consume"
into a narrow implementation track.

## Roadmap Source

- `docs/ecosystem/module_incubation_strategy.md`
- `specs/ecosystem/README.md`
- Ecosystem incubation follow-up: minimal adoption-trajectory fixture for `lifecourse`

## Functional Requirements

1. Define a portable fixture manifest for adoption trajectories with
   `scenario_id`, `intervention_id`, `time`, `adoption`,
   `cumulative_adoption`, `population`, `segment`, and `uncertainty_label`.
2. Store tabular fixture payloads in Arrow-compatible or Parquet-compatible
   form, with a small JSON manifest for metadata and provenance.
3. Document producer and consumer responsibilities for `innovate` and
   `lifecourse`.
4. Add schema-version and compatibility metadata that can be checked without
   importing either project's private Python objects.
5. Add fixture validation that proves the artifact can be loaded and checked
   from a base `innovate` install.

## Non-Functional Requirements

1. The fixture must not add `lifecourse` as a base dependency.
2. The fixture must stay deterministic and small enough for CI.
3. The contract must remain binding-friendly and aligned with the Arrow
   interchange direction.
4. Any future adapter promotion must remain behind optional extras, smoke CI,
   and compatibility matrices.

## Acceptance Criteria

1. A versioned adoption-trajectory fixture and manifest exist under
   `specs/ecosystem/`.
2. Tests validate the manifest, required columns, schema version, and
   deterministic row count.
3. Documentation explains how `lifecourse` should consume the fixture without
   importing `innovate` internals.
4. The ecosystem docs link this fixture to the documented adapter promotion
   policy.

## Out of Scope

1. Implementing a runtime `lifecourse` adapter.
2. Adding `lifecourse` as a dependency.
3. Defining all future HEOR scenario artifacts.
4. Publishing a cross-repository compatibility matrix beyond the first fixture.
