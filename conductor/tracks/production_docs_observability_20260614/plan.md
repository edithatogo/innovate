# Implementation Plan

## Phase 1: Production Docs Gates [checkpoint: 3a432bd]

- [x] Task: Add production docs verification contract [592dd0b]
    - [x] Define route, redirect, sitemap, search, version, and API generation checks
    - [x] Add tests that fail when production docs evidence is missing or stale
    - [x] Document local and CI verification commands
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Finalize DocSearch gate behavior [26ad957]
    - [x] Verify fallback behavior without Algolia credentials
    - [x] Add deployment-secret documentation and CI-safe checks
    - [x] Add evidence fields for enabled, disabled, or externally blocked status
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Production Docs Gates' (Protocol in workflow.md) [3a432bd]

## Phase 2: Evidence-Backed Dashboards [checkpoint: 19bd9aa]

- [x] Task: Add release and maturity dashboard artifacts [b1d0848]
    - [x] Generate docs data from release-readiness, Rust ownership, registry, and binding conformance artifacts
    - [x] Add Starlight pages that render status without duplicated claims
    - [x] Add stale data tests
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add observability and maintenance pages [e12c412]
    - [x] Add package health, compatibility, deprecation, support, and maintenance policy pages
    - [x] Link dashboards to machine-readable evidence
    - [x] Validate route coverage and links
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Evidence-Backed Dashboards' (Protocol in workflow.md) [19bd9aa]

## Phase 3: Examples, API Snippets, and Deployment Readiness

- [x] Task: Validate examples and API snippets [55a3fe8]
    - [x] Add or update snippet validation for Python and binding examples
    - [x] Classify examples that require optional dependencies or external credentials
    - [x] Add CI evidence for runnable examples
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add production deployment readiness evidence [b1c2a67]
    - [x] Verify GitHub Pages workflow, generated routes, and deployment artifacts
    - [x] Add docs release checklist and rollback notes
    - [x] Refresh Starlight validation evidence
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Examples, API Snippets, and Deployment Readiness' (Protocol in workflow.md)
