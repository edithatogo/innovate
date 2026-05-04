# Implementation Plan: Voiage Diffusion-Uncertainty Fixture

## Phase 1: Uncertainty Fixture Contract

- [x] Task: Define the uncertainty schema
    - [x] Specify parameter-draw, trajectory, scenario, and provenance fields
    - [x] Map fixture fields to VOI concepts without depending on `voiage`
    - [x] Confirm Arrow-compatible or Parquet-compatible payload expectations
- [x] Task: Add fixture-contract tests
    - [x] Write failing tests for required uncertainty fields and schema metadata
    - [x] Check that fixture loading does not require `voiage`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Uncertainty Fixture Contract' (Protocol in workflow.md)

## Phase 2: Fixture Implementation

- [x] Task: Add the deterministic fixture
    - [x] Create the fixture manifest under `specs/ecosystem/`
    - [x] Add compact tabular payloads for parameter draws and trajectories
    - [x] Record provenance, sample dimensions, and compatibility notes
- [x] Task: Validate local fixture consumption
    - [x] Run focused fixture tests
    - [x] Confirm base-install validation works without optional sibling dependencies
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Fixture Implementation' (Protocol in workflow.md)

## Phase 3: Documentation And Promotion Gates

- [x] Task: Update ecosystem documentation
    - [x] Link the fixture from ecosystem docs and roadmap-gap notes
    - [x] Document the adapter promotion stage and future compatibility-matrix requirement
    - [x] Explain that VOI method implementation remains owned outside `innovate`
- [x] Task: Run validation gates
    - [x] Run focused ecosystem fixture tests
    - [x] Run relevant docs or prose checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation And Promotion Gates' (Protocol in workflow.md)
