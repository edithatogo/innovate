# Model Validation and Benchmark Expansion Plan

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Benchmark Gap Audit

- [x] Task: Inventory benchmark coverage
    - [x] Compare benchmark corpus and model cards against policy, competition, substitution, network, multi-product, and causal surfaces.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write benchmark coverage tests
    - [x] Add tests requiring benchmark/model-card coverage or explicit rationale for every promoted family.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Gap Audit' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Validation Artifact Implementation

- [x] Task: Implement validation reports
    - [x] Add residual, out-of-sample, sensitivity, uncertainty, and calibration artifacts.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement leaderboard artifacts
    - [x] Add schema-tested benchmark comparison artifacts with reproducibility metadata.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Validation Artifact Implementation' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Corpus Expansion and Docs

- [x] Task: Add benchmark cases
    - [x] Add fast metadata benchmark cases for promoted policy, competition, substitution, and network surfaces.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Update Starlight docs and release evidence
    - [x] Document benchmark interpretation and wire evidence into release readiness.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Corpus Expansion and Docs' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: pending]

## Phase 4: Review, Push, and CI

- [x] Task: Run benchmark validation
    - [x] Run benchmark metadata tests, targeted model tests, and `uv run nox -s lint types tests docs package`.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run conductor-review, push, and monitor CI
    - [x] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
