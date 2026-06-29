# Scenario Experiment Workflows Plan

## Phase 1: Scenario Contract Schemas

## Phase 1 Checkpoint: [checkpoint: pending] Scenario Contract

- [ ] Task: Define scenario schemas
    - [ ] Add tests for baseline, intervention, substitution, competition, and network scenario specs.
    - [ ] Implement validated schema models and artifact envelopes.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add scenario registry metadata
    - [ ] Register scenario payload families in capability/model metadata.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Scenario Contract' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Experiment Runner

- [ ] Task: Implement scenario execution
    - [ ] Add runner APIs for scenario grids, deterministic seeds, diagnostics, and artifact export.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement scenario comparison summaries
    - [ ] Add ranking, incremental effect, threshold timing, and uncertainty summaries.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Experiment Runner' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Docs, Examples, and Release Evidence

- [ ] Task: Add examples and Starlight tutorials
    - [ ] Document policy, competition, and substitution scenario workflows.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add release and binding evidence
    - [ ] Record JSON/Arrow compatibility and Rust/polyglot status for scenario payloads.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Docs, Examples, and Release Evidence' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: pending]

## Phase 4: Review, Push, and CI

- [ ] Task: Run full scenario validation
    - [ ] Run targeted tests plus `uv run nox -s lint types tests docs package`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review, push, and monitor CI
    - [ ] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
