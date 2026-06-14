# Implementation Plan

## Phase 1: Release Gate Inventory and Fail-Closed Tests

- [ ] Task: Define the mature release gate contract
    - [ ] Inventory required release, security, docs, Rust, and binding checks
    - [ ] Define machine-readable release-readiness status values
    - [ ] Add tests that fail when required evidence is missing or stale
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Add release-readiness report generation
    - [ ] Implement a local report command or nox session
    - [ ] Emit JSON and human-readable summaries
    - [ ] Document how maintainers interpret release candidate versus release-ready states
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Release Gate Inventory and Fail-Closed Tests' (Protocol in workflow.md)

## Phase 2: Security, Provenance, and Reproducibility

- [ ] Task: Add supply-chain evidence gates
    - [ ] Add SBOM, dependency audit, license inventory, and checksum evidence generation
    - [ ] Add CI checks that verify evidence freshness
    - [ ] Ensure local checks do not require secrets
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Add reproducibility gates
    - [ ] Add deterministic benchmark fixture checks
    - [ ] Add seeded simulation and generated artifact reproducibility checks
    - [ ] Record acceptable nondeterminism with owner and rationale
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Security, Provenance, and Reproducibility' (Protocol in workflow.md)

## Phase 3: CI Enforcement and Release Dry Run

- [ ] Task: Add GitHub Actions release-readiness workflow
    - [ ] Wire the local release-readiness command into CI
    - [ ] Split slow checks into release or scheduled lanes without weakening required fast checks
    - [ ] Add artifact upload for readiness reports
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Execute and document a release dry run
    - [ ] Run package build checks for all package surfaces
    - [ ] Verify docs and registry receipts consume the readiness artifact
    - [ ] Update release documentation with the final gate sequence
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'CI Enforcement and Release Dry Run' (Protocol in workflow.md)
