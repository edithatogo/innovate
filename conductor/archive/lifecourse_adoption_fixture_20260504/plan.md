# Implementation Plan: Lifecourse Adoption-Trajectory Fixture

## Phase 1: Fixture Contract

- [x] Task: Define the adoption-trajectory schema
    - [x] Specify required columns, types, schema version, and provenance fields
    - [x] Define producer and consumer responsibilities for `innovate` and `lifecourse`
    - [x] Confirm the fixture remains compatible with Arrow or Parquet interchange
- [x] Task: Add fixture-contract tests
    - [x] Write failing tests for the manifest, required columns, and schema version
    - [x] Check that fixture loading does not require `lifecourse`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Fixture Contract' (Protocol in workflow.md)

## Phase 2: Fixture Implementation

- [x] Task: Add the deterministic fixture
    - [x] Create the fixture manifest under `specs/ecosystem/`
    - [x] Add the tabular payload in an Arrow-compatible or Parquet-compatible format
    - [x] Record provenance, fixture purpose, and compatibility notes
- [x] Task: Validate local fixture consumption
    - [x] Run focused fixture tests
    - [x] Confirm the base install can inspect the fixture without optional sibling dependencies
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Fixture Implementation' (Protocol in workflow.md)

## Phase 3: Documentation And Promotion Gates

- [x] Task: Update ecosystem documentation
    - [x] Link the fixture from ecosystem docs and roadmap-gap notes
    - [x] Document the adapter promotion stage and future compatibility-matrix requirement
    - [x] Explain that runtime adapter implementation remains future work
- [x] Task: Run validation gates
    - [x] Run focused ecosystem fixture tests
    - [x] Run relevant docs or prose checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation And Promotion Gates' (Protocol in workflow.md)
