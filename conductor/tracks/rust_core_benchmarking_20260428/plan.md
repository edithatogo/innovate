# Implementation Plan: Rust Core Benchmarking and Profiling Tooling

## Phase 1: Rust Benchmark Harness

- [~] Task: Define benchmark coverage for the native Rust kernel paths
    - [~] Select the native operations that should be benchmarked first
    - [~] Define stable benchmark inputs and measurements
    - [~] Document the benchmark gating expectations
- [ ] Task: Add a Rust benchmark harness
    - [ ] Add a `criterion`-based benchmark setup to the Rust crate
    - [ ] Add benchmark cases for the native logistic execution paths
    - [ ] Ensure the harness stays focused on native Rust execution
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Benchmark Harness' (Protocol in workflow.md)

## Phase 2: Rust Profiling Workflow

- [ ] Task: Add a native Rust profiling workflow
    - [ ] Document how to profile native Rust hot paths locally
    - [ ] Add a repeatable profiling command or script for the Rust crate
    - [ ] Keep the workflow separate from the Python Scalene setup
- [ ] Task: Update docs and governance to reflect Rust performance tooling
    - [ ] Update the Rust README with benchmark and profiling instructions
    - [ ] Update the Rust core roadmap with the benchmark/profiling tooling direction
    - [ ] Update the tech stack to record the Rust performance tooling choices
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Rust Profiling Workflow' (Protocol in workflow.md)
