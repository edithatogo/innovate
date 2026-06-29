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

## Phase 2 Checkpoint: [checkpoint: d02640c]

## Phase 3: Benchmark and Release Evidence

- [x] Task: Verify Rust benchmark and profiling evidence (d02640c)
    - [x] Cargo builds successfully with no Rust compilation errors.
    - [x] Ownership inventory backed by promotion evidence from completed Rust core tracks.
- [x] Task: Verify release and docs claims (d02640c)
    - [x] README, Starlight docs, and Rust core evidence verified as aligned with ownership inventory.
- [x] Task: Phase 3 Verification (d02640c)

## Phase 3 Checkpoint: [checkpoint: d02640c]

## Phase 4: Final Validation and Release

- [x] Task: Run full Python validation (d02640c)
    - [x] Run full test suite covering all ownership assertions.
    - [x] Verify all 29 ownership tests pass.
- [x] Task: Final review and push (d02640c)
    - [x] Run final conductor-review, apply findings.
    - [x] Push and monitor GitHub Actions.
- [x] Task: Phase 4 Verification (d02640c)

## Phase 4 Checkpoint: [checkpoint: d02640c]