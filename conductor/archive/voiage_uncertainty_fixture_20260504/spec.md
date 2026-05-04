# Specification: Voiage Diffusion-Uncertainty Fixture

## Overview

Define the first deterministic diffusion-uncertainty fixture that `voiage` can
use for HEOR Value of Information examples without importing `innovate`
internals. This track converts the ecosystem incubation follow-up "Define a
diffusion-uncertainty fixture that `voiage` can use for VOI examples" into a
narrow implementation track.

## Roadmap Source

- `docs/ecosystem/module_incubation_strategy.md`
- `specs/ecosystem/README.md`
- Ecosystem incubation follow-up: diffusion-uncertainty fixture for `voiage`

## Functional Requirements

1. Define a portable uncertainty fixture with parameter draws, adoption
   trajectories, uncertainty labels, scenario identifiers, and provenance.
2. Include metadata that maps uncertainty dimensions to VOI concepts without
   requiring `voiage` runtime objects.
3. Align tabular payloads with Arrow-compatible or Parquet-compatible
   interchange and small JSON manifests.
4. Include deterministic fixture values suitable for documentation, tests, and
   downstream VOI examples.
5. Add validation that checks schema version, required fields, and stable
   sample dimensions.

## Non-Functional Requirements

1. The fixture must not add `voiage` as a base dependency.
2. The fixture must avoid pickle or private Python object contracts.
3. The contract must stay usable from Python and future language bindings.
4. The fixture must be small enough for normal unit-test and docs workflows.

## Acceptance Criteria

1. A versioned diffusion-uncertainty fixture and manifest exist under
   `specs/ecosystem/`.
2. Tests validate schema metadata, uncertainty dimensions, deterministic
   fixture size, and base-install loading.
3. Documentation explains how `voiage` can consume the fixture as a
   decision-relevant uncertainty source.
4. The ecosystem docs keep runtime adapter implementation and VOI method
   implementation out of scope for this fixture track.

## Out of Scope

1. Implementing VOI methods.
2. Adding `voiage` as a dependency.
3. Implementing a runtime cross-repository adapter.
4. Defining every possible uncertainty payload used by future VOI workflows.
