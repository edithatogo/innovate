# Implementation Plan: Rust Core Benchmarking and Profiling Tooling

## Phase 1: Rust Benchmark Harness

- [x] Task: Define benchmark coverage for the native Rust kernel paths
    - [x] Select the native operations that should be benchmarked first
    - [x] Define stable benchmark inputs and measurements
    - [x] Document the benchmark gating expectations
- [x] Task: Add a Rust benchmark harness
    - [x] Add a `criterion`-based benchmark setup to the Rust crate
    - [x] Add benchmark cases for the native logistic execution paths
    - [x] Ensure the harness stays focused on native Rust execution
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Benchmark Harness' (Protocol in workflow.md)

## Phase 2: Rust Profiling Workflow

- [x] Task: Add a native Rust profiling workflow
    - [x] Document how to profile native Rust hot paths locally
    - [x] Add a repeatable profiling command or script for the Rust crate
    - [x] Keep the workflow separate from the Python Scalene setup
- [x] Task: Update docs and governance to reflect Rust performance tooling
    - [x] Update the Rust README with benchmark and profiling instructions
    - [x] Update the Rust core roadmap with the benchmark/profiling tooling direction
    - [x] Update the tech stack to record the Rust performance tooling choices
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Rust Profiling Workflow' (Protocol in workflow.md)
