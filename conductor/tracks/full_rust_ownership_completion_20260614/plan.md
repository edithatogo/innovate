# Implementation Plan

## Phase 1: Ownership Ledger and Contract Tests [checkpoint: cfe08e0]

- [x] Task: Build the final Rust ownership ledger `abaa869`
    - [x] Inventory every canonical operation, model family, and stable payload shape from current Python, Rust, and binding registries
    - [x] Classify each item as promote-now, retain-outside-core, or requires-design-decision
    - [x] Add machine-readable owner, rationale, evidence path, and release-claim state
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add fail-first ownership contract tests `bf8acbc`
    - [x] Add tests that fail for unowned stable payload shapes
    - [x] Add tests that fail for promoted model families missing Rust operation coverage
    - [x] Add tests that fail when docs claim full ownership without evidence
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Ownership Ledger and Contract Tests' (Protocol in workflow.md) `cfe08e0`

## Phase 2: Rust-Native Promotion

- [x] Task: Promote deterministic bridge model families `feeae52`
    - [x] Promote the feasible deterministic Norton-Bass single-generation fit slice
    - [x] Keep composite and multi-product deterministic slices explicit design-boundary items until stable schemas exist
    - [x] Expose fit, predict, summarize, diagnostics, and serialization where mathematically stable
    - [x] Add Python parity tests against existing reference behavior
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Promote stable payload shapes `f39953e`
    - [x] Add schema-backed Rust payload support for stable fitted-state, deterministic diagnostics, deterministic simulation, and simple positive observation fit payloads
    - [x] Keep covariate and event split payloads as explicit design-boundary exclusions until stable model-family schemas exist
    - [x] Add round-trip tests through Rust bindings and artifact contract tests for ownership evidence
    - [x] Preserve explicit exclusions for posterior, graph, agent, callback-heavy, stochastic, covariate, and event payloads
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
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
