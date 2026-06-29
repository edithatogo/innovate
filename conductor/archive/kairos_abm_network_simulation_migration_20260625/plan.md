# Kairos ABM and Network Simulation Migration Plan

## Phase 0: Kairos Inclusion Prerequisite

- [x] Task: Verify dependency inclusion prerequisite
    - [x] Confirm `kairos_dependency_inclusion_20260626` has completed or record its blocker before starting behavior-level adapter work.
    - [x] Confirm release evidence identifies the Kairos repository revision or published crate versions, selected crate set, build status, smoke status, and bridge-crate status.
    - [x] Confirm Mesa and NDLib are no longer required for the base install or record the external compatibility constraint that blocks removal.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 0: Kairos Inclusion Prerequisite' (Protocol in workflow.md)

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Adapter Audit and Contract

- [x] Task: Audit simulation surfaces
    - [x] Inventory ABM, NDLib, network, policy, examples, dependency evidence, and docs after the Kairos inclusion prerequisite is satisfied.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Define Kairos adapter contract
    - [x] Write failing tests for deterministic scheduler events, ECS-style agent state, DES trajectory/resource queue events, ABM behavior updates, deterministic random streams, Arrow/JSON telemetry artifacts, topology, interventions, seeds, and policy/network diffusion traces.
    - [x] Implement validated contract models and dependency evidence checks.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Adapter Audit and Contract' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Adapter Implementation and Migration

- [x] Task: Implement Kairos adapter path
    - [x] Add adapter wiring, fallback diagnostics, and installed-path smoke tests using the Kairos source and bridge-crate promotion status established by `kairos_dependency_inclusion_20260626`.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Preserve legacy compatibility boundaries
    - [x] Add migration notes and fail-safe behavior for legacy ABM examples without restoring Mesa or NDLib as base runtime requirements.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Adapter Implementation and Migration' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Docs, Evidence, and CI

- [x] Task: Update Starlight docs and model cards
    - [x] Document Kairos policy, simulation examples, and artifact schemas.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run review, push, and CI monitor
    - [x] Run targeted tests, `uv run nox -s lint types tests docs package`, conductor-review, push, and monitor GitHub Actions.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Docs, Evidence, and CI' (Protocol in workflow.md)
