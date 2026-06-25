# Kairos ABM and Network Simulation Migration Plan

## Phase 0: Kairos Inclusion Prerequisite

- [ ] Task: Verify dependency inclusion prerequisite
    - [ ] Confirm `kairos_dependency_inclusion_20260626` has completed or record its blocker before starting behavior-level adapter work.
    - [ ] Confirm release evidence identifies the Kairos repository revision or published crate versions, selected crate set, build status, smoke status, and bridge-crate status.
    - [ ] Confirm Mesa and NDLib are no longer required for the base install or record the external compatibility constraint that blocks removal.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 0: Kairos Inclusion Prerequisite' (Protocol in workflow.md)

## Phase 1: Adapter Audit and Contract

- [ ] Task: Audit simulation surfaces
    - [ ] Inventory ABM, NDLib, network, policy, examples, dependency evidence, and docs after the Kairos inclusion prerequisite is satisfied.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Define Kairos adapter contract
    - [ ] Write failing tests for deterministic scheduler events, ECS-style agent state, DES trajectory/resource queue events, ABM behavior updates, deterministic random streams, Arrow/JSON telemetry artifacts, topology, interventions, seeds, and policy/network diffusion traces.
    - [ ] Implement validated contract models and dependency evidence checks.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Adapter Audit and Contract' (Protocol in workflow.md)

## Phase 2: Adapter Implementation and Migration

- [ ] Task: Implement Kairos adapter path
    - [ ] Add adapter wiring, fallback diagnostics, and installed-path smoke tests using the Kairos source and bridge-crate promotion status established by `kairos_dependency_inclusion_20260626`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Preserve legacy compatibility boundaries
    - [ ] Add migration notes and fail-safe behavior for legacy ABM examples without restoring Mesa or NDLib as base runtime requirements.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Adapter Implementation and Migration' (Protocol in workflow.md)

## Phase 3: Docs, Evidence, and CI

- [ ] Task: Update Starlight docs and model cards
    - [ ] Document Kairos policy, simulation examples, and artifact schemas.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run review, push, and CI monitor
    - [ ] Run targeted tests, `uv run nox -s lint types tests docs package`, conductor-review, push, and monitor GitHub Actions.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Docs, Evidence, and CI' (Protocol in workflow.md)
