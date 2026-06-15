# Implementation Plan

## Phase 1: Live Evidence Refresh [checkpoint: 0045896]

- [x] Task: Refresh package-manager receipt evidence [f52cc66]
    - [x] Verify PyPI, npm, crates.io, Julia General, Go modules, NuGet, and R-universe current states
    - [x] Record version, receipt URL, acceptance state, and evidence timestamp
    - [x] Add drift tests for stale or mismatched registry states
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Refresh pending external target evidence [8dcf40b]
    - [x] Re-check CRAN, Spack, EasyBuild, HPSF, E4S, and community target requirements
    - [x] Update blocker, owner, and next-action fields with current evidence
    - [x] Remove any generic blocked language
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Live Evidence Refresh' (Protocol in workflow.md)

## Phase 2: Submission Packet Completion

- [x] Task: Complete CRAN and scientific submission packets [e4438d7]
    - [x] Refresh R package checks, CRAN comments, and source package evidence
    - [x] Prepare pyOpenSci, rOpenSci, JOSS, NumFOCUS, Arrow, .NET, Julia, and R community packets
    - [x] Record exact maintainer action boundaries
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [~] Task: Complete HPC submission packets
    - [x] Refresh Spack and EasyBuild candidate recipes and scheduler evidence
    - [x] Prepare HPSF and E4S proposal/contact packets
    - [x] Record sponsor, contact, CI, and review expectations
    - [x] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Submission Packet Completion' (Protocol in workflow.md)

## Phase 3: Receipts, Deferrals, and Claim Closure

- [ ] Task: Record receipts or owner-backed deferrals
    - [ ] Store external receipts when submissions are actually made
    - [ ] Store maintainer-ready deferrals when final action remains external
    - [ ] Add tests that distinguish readiness, submission, acceptance, and deferral
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Reconcile registry docs and release claims
    - [ ] Update Sphinx archival sources and Starlight active docs
    - [ ] Refresh machine-readable inventories
    - [ ] Add stale-claim tests for accepted/submitted wording
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Receipts, Deferrals, and Claim Closure' (Protocol in workflow.md)
