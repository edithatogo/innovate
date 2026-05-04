# Implementation Plan: HEOML Schema Placement Decision

## Phase 1: Decision Analysis

- [x] Task: Compare schema placement options
    - [x] Assess repo-local `innovate`, embedded `lifecourse`, and standalone `heoml` homes
    - [x] Document ownership, versioning, compatibility, publication, and migration tradeoffs
    - [x] Identify the interim recommendation and migration trigger
- [x] Task: Add governance checks
    - [x] Write failing documentation checks for the selected schema home and migration trigger
    - [x] Check that private Python object contracts remain excluded
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Decision Analysis' (Protocol in workflow.md)

## Phase 2: Decision Documentation

- [x] Task: Add HEOML schema placement documentation
    - [x] Record the selected interim schema home
    - [x] Record versioning, compatibility, and deprecation expectations
    - [x] Cross-link ecosystem contracts and roadmap-gap notes
- [x] Task: Validate documentation
    - [x] Run focused documentation tests
    - [x] Run relevant prose checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Decision Documentation' (Protocol in workflow.md)
