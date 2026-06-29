# Rust-Native Ownership Release Proof Plan

## Phase 1: Ownership Audit

- [~] Task: Build operation and model-family matrix
    - [x] Inventory canonical operations, stable payloads, model families, and current Rust/native/fallback status.
    - [~] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write failing coverage tests
    - [ ] Add tests that require every stable operation and model family to have native or exception status.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Ownership Audit' (Protocol in workflow.md)

## Phase 2: Native Parity and Exceptions

- [ ] Task: Complete eligible Rust-native slices
    - [ ] Implement missing native operation/model slices where feasible.
    - [ ] Add golden fixture parity tests.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Promote explicit Python-reference exceptions
    - [ ] Record owner, rationale, fallback diagnostics, revisit condition, and release-claim wording for remaining exceptions.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Native Parity and Exceptions' (Protocol in workflow.md)

## Phase 3: Benchmark and Release Evidence

- [ ] Task: Refresh Rust benchmark and profiling evidence
    - [ ] Run cargo checks, no-run benches, and promotion benchmark smoke where local environment supports it.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update release and docs claims
    - [ ] Reconcile README, Starlight docs, release-readiness, and Rust core evidence with proven ownership.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmark and Release Evidence' (Protocol in workflow.md)

## Phase 4: Review, Push, and CI

- [ ] Task: Run full Rust and Python validation
    - [ ] Run `cargo test`, `cargo check --benches --examples`, and targeted Python parity tests.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run final conductor-review and CI monitor
    - [ ] Apply review findings, push, and monitor GitHub Actions until green or externally blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
