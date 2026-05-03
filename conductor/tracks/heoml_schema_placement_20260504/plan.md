# Implementation Plan: HEOML Schema Placement Decision

## Phase 1: Decision Analysis

- [ ] Task: Compare schema placement options
    - [ ] Assess repo-local `innovate`, embedded `lifecourse`, and standalone `heoml` homes
    - [ ] Document ownership, versioning, compatibility, publication, and migration tradeoffs
    - [ ] Identify the interim recommendation and migration trigger
- [ ] Task: Add governance checks
    - [ ] Write failing documentation checks for the selected schema home and migration trigger
    - [ ] Check that private Python object contracts remain excluded
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Decision Analysis' (Protocol in workflow.md)

## Phase 2: Decision Documentation

- [ ] Task: Add HEOML schema placement documentation
    - [ ] Record the selected interim schema home
    - [ ] Record versioning, compatibility, and deprecation expectations
    - [ ] Cross-link ecosystem contracts and roadmap-gap notes
- [ ] Task: Validate documentation
    - [ ] Run focused documentation tests
    - [ ] Run relevant prose checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Decision Documentation' (Protocol in workflow.md)
