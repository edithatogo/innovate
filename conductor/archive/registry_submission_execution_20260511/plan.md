# Implementation Plan: Registry Submission Execution and Receipt Capture

## Phase 1: Submission Inventory and Release Readiness [checkpoint: 795e9b5]

- [x] Task: Build a complete submission inventory c3bb3c9
    - [x] Enumerate every package-manager registry target
    - [x] Enumerate every HPC registry and packaging target
    - [x] Map each target to its package surface, owner, and release path
- [x] Task: Verify release prerequisites b78b20f
    - [x] Confirm package versions and names are aligned
    - [x] Confirm publication gates exist for each target
    - [x] Confirm credential or maintainer access requirements are known
- [x] Task: Write failing tests for submission-state tracking b78b20f
    - [x] Add tests that require each target to have a submission status
    - [x] Add tests that reject readiness-only language once a submission is complete
    - [x] Add tests that require an auditable receipt or blocker note for each target
- [x] Task: Conductor - Automated Review and Checkpoint 'Submission Inventory and Release Readiness' (Protocol in workflow.md)

## Phase 2: Execute Package-Manager Submissions [checkpoint: 79a4ddc]

- [x] Task: Submit the language packages through their registry paths
    - [x] Publish or attempt publish for PyPI/TestPyPI
    - [x] Publish or attempt publish for npm
    - [x] Publish or attempt publish for crates.io
    - [x] Publish or prepare R-universe/CRAN submission evidence
    - [x] Submit Julia General registry metadata
    - [x] Tag and verify Go module release flow
    - [x] Publish or dry-run NuGet release artifacts
- [x] Task: Capture package-registry receipts
    - [x] Record registry URLs, version records, or submission IDs
    - [x] Record logs for any blocked or deferred submissions
    - [x] Preserve package-manager artifacts in the evidence bundle
- [x] Task: Conductor - Automated Review and Checkpoint 'Execute Package-Manager Submissions' (Protocol in workflow.md)

## Phase 3: Execute HPC Registry Submissions [checkpoint: 991247c]

- [x] Task: Exercise the HPC packaging candidates in submission mode
    - [x] Validate the Spack recipe in a scheduler-backed environment
    - [x] Validate the EasyBuild easyconfig in a scheduler-backed environment
    - [x] Gather any HPSF review or registration evidence
    - [x] Gather any E4S review or registration evidence
- [x] Task: Capture HPC registry receipts
    - [x] Record scheduler metadata and batch logs
    - [x] Record HPC registry links, review IDs, or blocker notes
    - [x] Update the HPC evidence bundle with submission outcomes
    - Local execution evidence is captured in `docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json`, `docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log`, the captured Slurm/PBS batch logs, and the HPSF/E4S blocker notes.
- [x] Task: Conductor - Automated Review and Checkpoint 'Execute HPC Registry Submissions' (Protocol in workflow.md)

## Phase 4: Reconcile Docs and Status Matrices [checkpoint: 991247c]

- [x] Task: Update registry-facing documentation
    - [x] Mark submitted targets as submitted instead of merely ready
    - [x] Mark deferred targets with explicit blocker reasons
    - [x] Update the binding, HPC, and registry plan docs with actual outcomes
- [x] Task: Update machine-readable status fixtures
    - [x] Refresh submission matrices and readiness tables
    - [x] Add durable links to receipts and registry URLs
    - [x] Keep the contract docs aligned with the actual submission state
- [x] Task: Add regression tests for the reconciled status
    - [x] Test that the docs reference the recorded receipts
    - [x] Test that no page overstates submission state
    - [x] Test that registry statuses stay synchronized with the evidence bundle
- [x] Task: Conductor - Automated Review and Checkpoint 'Reconcile Docs and Status Matrices' (Protocol in workflow.md)

## Phase 5: Final Review and Archive [checkpoint: 923223a]

- [x] Task: Run final conductor review
    - [x] Review the full submission diff against the spec, plan, workflow, and tests
    - [x] Apply any high-confidence fixes surfaced by the review
    - [x] Re-run validation until the track is stable
- [x] Task: Archive the completed submission track
    - [x] Move the track folder to the archive location
    - [x] Update the tracks registry entry to completed
    - [x] Preserve links to the final submission evidence bundle
- [x] Task: Conductor - Automated Review and Checkpoint 'Final Review and Archive' (Protocol in workflow.md)
