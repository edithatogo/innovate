# Implementation Plan: Julia Installed-Package Bridge Runtime Readiness

## Phase 1: Runtime Path Fix

- [x] Task: Separate checkout and installed-package runtime paths in Julia
    - [x] Add a repo-root detector that only succeeds in checkout mode
    - [x] Keep checkout-mode bridge behavior intact
    - [x] Make installed-package bridge calls avoid repo-relative Python path assumptions
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Runtime Path Fix' (Protocol in workflow.md)

## Phase 2: Installed-Package Validation

- [x] Task: Add installed-package smoke coverage and workflow wiring
    - [x] Add a smoke test for installed-package Julia bridge usage
    - [x] Update CI or publish workflows to run the smoke step
    - [x] Update Julia docs to describe the installed-package runtime contract
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Installed-Package Validation' (Protocol in workflow.md)
