# Implementation Plan: Registry Submission Execution and Receipt Capture

## Phase 1: Submission Inventory and Release Readiness

- [x] Task: Build a complete submission inventory c3bb3c9
    - [x] Enumerate every package-manager registry target
    - [x] Enumerate every HPC registry and packaging target
    - [x] Map each target to its package surface, owner, and release path
- [~] Task: Verify release prerequisites
    - [ ] Confirm package versions and names are aligned
    - [ ] Confirm publication gates exist for each target
    - [ ] Confirm credential or maintainer access requirements are known
- [ ] Task: Write failing tests for submission-state tracking
    - [ ] Add tests that require each target to have a submission status
    - [ ] Add tests that reject readiness-only language once a submission is complete
    - [ ] Add tests that require an auditable receipt or blocker note for each target
- [ ] Task: Conductor - Automated Review and Checkpoint 'Submission Inventory and Release Readiness' (Protocol in workflow.md)

## Phase 2: Execute Package-Manager Submissions

- [ ] Task: Submit the language packages through their registry paths
    - [ ] Publish or attempt publish for PyPI/TestPyPI
    - [ ] Publish or attempt publish for npm
    - [ ] Publish or attempt publish for crates.io
    - [ ] Publish or prepare R-universe/CRAN submission evidence
    - [ ] Submit Julia General registry metadata
    - [ ] Tag and verify Go module release flow
    - [ ] Publish or dry-run NuGet release artifacts
- [ ] Task: Capture package-registry receipts
    - [ ] Record registry URLs, version records, or submission IDs
    - [ ] Record logs for any blocked or deferred submissions
    - [ ] Preserve package-manager artifacts in the evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Execute Package-Manager Submissions' (Protocol in workflow.md)

## Phase 3: Execute HPC Registry Submissions

- [ ] Task: Exercise the HPC packaging candidates in submission mode
    - [ ] Validate the Spack recipe in a scheduler-backed environment
    - [ ] Validate the EasyBuild easyconfig in a scheduler-backed environment
    - [ ] Gather any HPSF review or registration evidence
    - [ ] Gather any E4S review or registration evidence
- [ ] Task: Capture HPC registry receipts
    - [ ] Record scheduler metadata and batch logs
    - [ ] Record HPC registry links, review IDs, or blocker notes
    - [ ] Update the HPC evidence bundle with submission outcomes
- [ ] Task: Conductor - Automated Review and Checkpoint 'Execute HPC Registry Submissions' (Protocol in workflow.md)

## Phase 4: Reconcile Docs and Status Matrices

- [ ] Task: Update registry-facing documentation
    - [ ] Mark submitted targets as submitted instead of merely ready
    - [ ] Mark deferred targets with explicit blocker reasons
    - [ ] Update the binding, HPC, and registry plan docs with actual outcomes
- [ ] Task: Update machine-readable status fixtures
    - [ ] Refresh submission matrices and readiness tables
    - [ ] Add durable links to receipts and registry URLs
    - [ ] Keep the contract docs aligned with the actual submission state
- [ ] Task: Add regression tests for the reconciled status
    - [ ] Test that the docs reference the recorded receipts
    - [ ] Test that no page overstates submission state
    - [ ] Test that registry statuses stay synchronized with the evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Reconcile Docs and Status Matrices' (Protocol in workflow.md)

## Phase 5: Final Review and Archive

- [ ] Task: Run final conductor review
    - [ ] Review the full submission diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes surfaced by the review
    - [ ] Re-run validation until the track is stable
- [ ] Task: Archive the completed submission track
    - [ ] Move the track folder to the archive location
    - [ ] Update the tracks registry entry to completed
    - [ ] Preserve links to the final submission evidence bundle
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Review and Archive' (Protocol in workflow.md)
