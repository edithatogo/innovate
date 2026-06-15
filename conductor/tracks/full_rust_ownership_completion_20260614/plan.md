# Implementation Plan

## Phase 1: Ownership Ledger and Contract Tests

- [x] Task: Build the final Rust ownership ledger `abaa869`
    - [x] Inventory every canonical operation, model family, and stable payload shape from current Python, Rust, and binding registries
    - [x] Classify each item as promote-now, retain-outside-core, or requires-design-decision
    - [x] Add machine-readable owner, rationale, evidence path, and release-claim state
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [~] Task: Add fail-first ownership contract tests
    - [ ] Add tests that fail for unowned stable payload shapes
    - [ ] Add tests that fail for promoted model families missing Rust operation coverage
    - [ ] Add tests that fail when docs claim full ownership without evidence
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Ownership Ledger and Contract Tests' (Protocol in workflow.md)

## Phase 2: Rust-Native Promotion

- [ ] Task: Promote deterministic bridge model families
    - [ ] Implement Rust-native kernels for feasible composite and multi-product deterministic slices
    - [ ] Expose fit, predict, summarize, diagnostics, and serialization where mathematically stable
    - [ ] Add Python parity tests against existing reference behavior
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Promote stable payload shapes
    - [ ] Add schema-backed Rust payload support for stable covariate, event split, fitted-state, and deterministic simulation payloads
    - [ ] Add round-trip tests through Python and Rust bindings
    - [ ] Preserve explicit exclusions for posterior, graph, agent, and callback-heavy payloads
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Rust-Native Promotion' (Protocol in workflow.md)

## Phase 3: Binding Dispatch, Benchmarks, and Claim Closure

- [ ] Task: Wire promoted Rust operations through binding dispatch
    - [ ] Update Python dispatch and binding smoke surfaces for promoted operations
    - [ ] Verify R, Julia, TypeScript, Go, and C# clients expose consistent capabilities or explicit non-support metadata
    - [ ] Add capability-registry drift tests
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Add benchmark and release-claim evidence
    - [ ] Add benchmark evidence for promoted native slices
    - [ ] Refresh ownership validation artifacts and roadmap wording
    - [ ] Add tests that reject stale future-state or overclaim wording
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Binding Dispatch, Benchmarks, and Claim Closure' (Protocol in workflow.md)
