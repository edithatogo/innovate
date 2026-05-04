# Implementation Plan: Release Notes and R Artifact Documentation Synchronization

## Phase 1: Red-Phase Release Documentation Guards

- [x] Task: Add release-note and R artifact drift tests
    - [x] Require current package version coverage in `CHANGELOG.md`
    - [x] Require a documented release-notes policy page
    - [x] Require R publication docs to match the source vignette and manual artifact
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Red-Phase Release Documentation Guards' (Protocol in workflow.md)

## Phase 2: Release Documentation Synchronization

- [x] Task: Update release-note policy and changelog
    - [x] Add the Sphinx release-notes policy page
    - [x] Add `0.4.0` and `0.5.0` changelog sections
    - [x] Link Release Drafter configuration to the policy
- [x] Task: Correct R artifact publication docs
    - [x] Replace stale no-vignette prose
    - [x] Document source vignette behavior and versioned R manual artifacts
- [x] Task: Validate and archive
    - [x] Run targeted policy tests
    - [x] Run docs and lint validation
    - [x] Normalize completed Conductor archive status text discovered during the governance audit
    - [x] Archive the completed Conductor track
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Release Documentation Synchronization' (Protocol in workflow.md)
