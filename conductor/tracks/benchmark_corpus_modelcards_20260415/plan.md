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
- [x] Task: Conductor - User Manual Verification 'Phase 2: Evaluation Harness' (Protocol in workflow.md) [359c788]

## Phase 3: Reporting and Documentation

- [x] Task: Generate model cards for stable models [359c788]
    - [x] Populate model-card content from implemented capabilities
    - [x] Validate that cards stay synchronized with the registry
    - [x] Add tests or checks for schema compliance
- [x] Task: Document benchmark workflows [359c788]
    - [x] Add contributor and user guidance for running the benchmarks
    - [x] Describe how to interpret outputs and compare models
    - [x] Verify all acceptance criteria are satisfied
- [x] Task: Conductor - User Manual Verification 'Phase 3: Reporting and Documentation' (Protocol in workflow.md) [359c788]
