# Implementation Plan: Benchmark Corpus and Model Cards

## Phase 1: Benchmark Definitions [checkpoint: f044011]

- [x] Task: Define the benchmark corpus [dc042f3]
    - [x] Select representative datasets and synthetic scenarios
    - [x] Assign stable identifiers and metadata for each benchmark case
    - [x] Write failing tests for dataset discovery and metadata integrity
- [x] Task: Define the model-card schema [dc042f3]
    - [x] Specify required fields for stable model families
    - [x] Add validation tests for model-card completeness
    - [x] Confirm the new tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Definitions' (Protocol in workflow.md)

## Phase 2: Evaluation Harness

- [x] Task: Implement the benchmark runner
    - [x] Add a canonical interface for running a model on a benchmark case
    - [x] Save standardized metrics and diagnostic outputs
    - [x] Make the harness tests pass
- [x] Task: Integrate stable model families
    - [x] Add runners for core diffusion, substitution, and competition models
    - [x] Verify reproducible outputs for smoke-test scenarios
    - [x] Add coverage for benchmark artifact generation
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Evaluation Harness' (Protocol in workflow.md)

## Phase 3: Reporting and Documentation

- [ ] Task: Generate model cards for stable models
    - [ ] Populate model-card content from implemented capabilities
    - [ ] Validate that cards stay synchronized with the registry
    - [ ] Add tests or checks for schema compliance
- [ ] Task: Document benchmark workflows
    - [ ] Add contributor and user guidance for running the benchmarks
    - [ ] Describe how to interpret outputs and compare models
    - [ ] Verify all acceptance criteria are satisfied
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reporting and Documentation' (Protocol in workflow.md)
