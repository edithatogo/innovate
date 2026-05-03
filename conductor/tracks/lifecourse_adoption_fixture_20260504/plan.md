# Implementation Plan: Lifecourse Adoption-Trajectory Fixture

## Phase 1: Fixture Contract

- [ ] Task: Define the adoption-trajectory schema
    - [ ] Specify required columns, types, schema version, and provenance fields
    - [ ] Define producer and consumer responsibilities for `innovate` and `lifecourse`
    - [ ] Confirm the fixture remains compatible with Arrow or Parquet interchange
- [ ] Task: Add fixture-contract tests
    - [ ] Write failing tests for the manifest, required columns, and schema version
    - [ ] Check that fixture loading does not require `lifecourse`
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Fixture Contract' (Protocol in workflow.md)

## Phase 2: Fixture Implementation

- [ ] Task: Add the deterministic fixture
    - [ ] Create the fixture manifest under `specs/ecosystem/`
    - [ ] Add the tabular payload in an Arrow-compatible or Parquet-compatible format
    - [ ] Record provenance, fixture purpose, and compatibility notes
- [ ] Task: Validate local fixture consumption
    - [ ] Run focused fixture tests
    - [ ] Confirm the base install can inspect the fixture without optional sibling dependencies
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Fixture Implementation' (Protocol in workflow.md)

## Phase 3: Documentation And Promotion Gates

- [ ] Task: Update ecosystem documentation
    - [ ] Link the fixture from ecosystem docs and roadmap-gap notes
    - [ ] Document the adapter promotion stage and future compatibility-matrix requirement
    - [ ] Explain that runtime adapter implementation remains future work
- [ ] Task: Run validation gates
    - [ ] Run focused ecosystem fixture tests
    - [ ] Run relevant docs or prose checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation And Promotion Gates' (Protocol in workflow.md)
