# Scenario Experiment Workflows Plan

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Scenario Contract

- [x] Task: Define scenario schemas
    - [x] Add tests for baseline, intervention, substitution, competition, and network scenario specs.
    - [x] Implement validated schema models and artifact envelopes.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add scenario registry metadata
    - [x] Register scenario payload families in capability/model metadata.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Scenario Contract' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Experiment Runner

- [x] Task: Implement scenario execution
    - [x] Add runner APIs for scenario grids, deterministic seeds, diagnostics, and artifact export.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement scenario comparison summaries
    - [x] Add ranking, incremental effect, threshold timing, and uncertainty summaries.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Experiment Runner' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Docs, Examples, and Release Evidence

- [x] Task: Add examples and Starlight tutorials
    - [x] Document policy, competition, and substitution scenario workflows.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add release and binding evidence
    - [x] Record JSON/Arrow compatibility and Rust/polyglot status for scenario payloads.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Docs, Examples, and Release Evidence' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: pending]

## Phase 4: Review, Push, and CI

- [x] Task: Run full scenario validation
    - [x] Run targeted tests plus `uv run nox -s lint types tests docs package`.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run conductor-review, push, and monitor CI
    - [x] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
