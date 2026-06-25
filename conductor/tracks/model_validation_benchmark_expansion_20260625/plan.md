# Model Validation and Benchmark Expansion Plan

## Phase 1: Benchmark Gap Audit

- [ ] Task: Inventory benchmark coverage
    - [ ] Compare benchmark corpus and model cards against policy, competition, substitution, network, multi-product, and causal surfaces.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write benchmark coverage tests
    - [ ] Add tests requiring benchmark/model-card coverage or explicit rationale for every promoted family.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Gap Audit' (Protocol in workflow.md)

## Phase 2: Validation Artifact Implementation

- [ ] Task: Implement validation reports
    - [ ] Add residual, out-of-sample, sensitivity, uncertainty, and calibration artifacts.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement leaderboard artifacts
    - [ ] Add schema-tested benchmark comparison artifacts with reproducibility metadata.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Validation Artifact Implementation' (Protocol in workflow.md)

## Phase 3: Corpus Expansion and Docs

- [ ] Task: Add benchmark cases
    - [ ] Add fast metadata benchmark cases for promoted policy, competition, substitution, and network surfaces.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update Starlight docs and release evidence
    - [ ] Document benchmark interpretation and wire evidence into release readiness.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Corpus Expansion and Docs' (Protocol in workflow.md)

## Phase 4: Review, Push, and CI

- [ ] Task: Run benchmark validation
    - [ ] Run benchmark metadata tests, targeted model tests, and `uv run nox -s lint types tests docs package`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review, push, and monitor CI
    - [ ] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
