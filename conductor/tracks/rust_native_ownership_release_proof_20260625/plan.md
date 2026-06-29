# Rust-Native Ownership Release Proof Plan

## Phase 1: Ownership Audit

- [x] Task: Build operation and model-family matrix (ac55f3d)
    - [x] Inventory canonical operations, stable payloads, model families, and current Rust/native/fallback status.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write failing coverage tests (ac55f3d)
    - [x] Add tests that require every stable operation and model family to have native or exception status.
    - [x] Commit task changes and attach the required Conductor git note.
- [~] Task: Conductor - Phase 1 Verification

## Phase 1 Checkpoint: [checkpoint: f1c4d85]

## Phase 2: Native Parity and Exceptions

- [x] Task: Complete eligible Rust-native slices (d83cd5a)
    - [x] Verify native operation/model slices are correctly documented.
    - [x] Bass, Logistic, Gompertz, FisherPry are fully native.
    - [x] Norton-Bass single-generation is native; multi-generation documented as exception.
    - [x] Parity tests exist for core native models.
- [x] Task: Promote explicit Python-reference exceptions (d83cd5a)
    - [x] Documented 5 exception models with owner, rationale, fallback diagnostics, revisit condition.
    - [x] Exception documentation is structured and complete.
    - [x] Commit task changes and attach the required Conductor git note.
- [~] Task: Conductor - Phase 2 Verification

## Phase 2 Checkpoint: [checkpoint: TBD]

## Phase 3: Benchmark and Release Evidence

- [~] Task: Verify Rust benchmark and profiling evidence
    - [~] Verify cargo builds, no Rust compilation errors detected.
    - [~] Ownership inventory backed by promotion evidence archives.
- [~] Task: Verify release and docs claims
    - [~] README, Starlight docs, and Rust core evidence aligned with ownership inventory.
- [~] Task: Conductor - Phase 3 Verification

## Phase 4 Checkpoint: [checkpoint: TBD]

## Phase 4: Final Validation and Release

- [~] Task: Run full Python validation
    - [~] Run full test suite covering all ownership assertions.
    - [~] Verify all 29 ownership tests pass.
- [~] Task: Final review and push
    - [~] Run final conductor-review, apply findings.
    - [~] Push and monitor GitHub Actions.
- [~] Task: Conductor - Phase 4 Verification
