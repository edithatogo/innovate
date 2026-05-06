# Implementation Plan

## Phase 1: Design the canonical version sync path

- [x] Task: Identify the canonical version source and sync targets
    - [x] Confirm `pyproject.toml` remains the release version source
    - [x] Enumerate the manifests that must match the canonical version
    - [x] Decide whether the checker should run in both read-only and write modes
- [x] Task: Conductor - Automated Review and Checkpoint 'Design the canonical version sync path' (Protocol in workflow.md)

## Phase 2: Implement the sync/check tool

- [x] Task: Add a repo version sync/check script
    - [x] Parse the canonical release version
    - [x] Compare package manifests against the canonical version
    - [x] Update the package manifests in write mode
    - [x] Exit non-zero in check mode when drift exists
- [x] Task: Add focused version-synchronization tests
    - [x] Verify the checker accepts aligned manifests
    - [x] Verify the checker fails on drift
    - [x] Verify write mode updates the supported files
- [x] Task: Conductor - Automated Review and Checkpoint 'Implement the sync/check tool' (Protocol in workflow.md)

## Phase 3: Wire the guard into docs and CI

- [x] Task: Add documentation for the versioning policy
    - [x] Explain the canonical source of truth
    - [x] Explain how maintainers run the sync tool locally
    - [x] Explain how CI prevents drift
- [x] Task: Add the sync/check guard to CI
    - [x] Run the checker in the Python CI gate
    - [x] Keep the existing release automation intact
- [x] Task: Conductor - Automated Review and Checkpoint 'Wire the guard into docs and CI' (Protocol in workflow.md)

## Phase 4: Close out the track

- [x] Task: Final review and archive readiness
    - [x] Confirm the versioning guard is documented
    - [x] Confirm the manifests are aligned
    - [x] Confirm the worktree is clean before archive
- [x] Task: Conductor - Automated Review and Checkpoint 'Close out the track' (Protocol in workflow.md)
