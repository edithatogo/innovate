# Implementation Plan: Quality Gates, CI, and Release Hardening

## Phase 1: Repair Local Quality Signals

- [x] Task: Audit current local quality-gate failures
    - [x] Reproduce current pytest, ruff, typing, packaging, and docs issues
    - [x] Categorize failures as config, optional dependency, or code defects
    - [x] Identify which commands should be part of the default contributor workflow
- [x] Task: Write failing regression tests or checks for broken gates
    - [x] Add tests or assertions for base test collection
    - [x] Add checks covering the intended type-check configuration
    - [x] Add packaging and docs smoke coverage where currently absent
- [x] Task: Implement local gate fixes
    - [x] Fix invalid or misleading tool configuration
    - [x] Make the documented local commands actually runnable
    - [x] Update contributor-facing docs for the canonical quality flow
- [x] Task: Conductor - User Manual Verification 'Phase 1: Repair Local Quality Signals' (Protocol in workflow.md)

## Phase 2: Harden CI and Release Workflows

- [x] Task: Make required CI jobs fail hard
    - [x] Remove masked failure paths from required jobs
    - [x] Separate optional jobs from gating jobs
    - [x] Ensure branch conditions and matrix behavior reflect active development policy
- [x] Task: Align packaging and release workflows
    - [x] Validate build and publish workflows against the current branch model
    - [x] Ensure release automation and changelog generation are coherent
    - [x] Review workflow permissions and reduce them where possible
- [x] Task: Add CI regression protection
    - [x] Add smoke checks for docs and package build
    - [x] Verify CI fails when a required quality gate fails
    - [x] Confirm acceptance criteria with workflow-level validation
- [x] Task: Conductor - User Manual Verification 'Phase 2: Harden CI and Release Workflows' (Protocol in workflow.md)

## Phase 3: Governance and Maturity Documentation

- [x] Task: Document repo maturity policy
    - [x] Add compatibility and deprecation guidance
    - [x] Document required checks for merge readiness
    - [x] Document how optional backends affect supported workflows
- [x] Task: Final verification and cleanup
    - [x] Re-run full local quality gates
    - [x] Verify CI and release documentation match reality
    - [x] Confirm the repo's quality story is internally consistent
- [x] Task: Conductor - User Manual Verification 'Phase 3: Governance and Maturity Documentation' (Protocol in workflow.md)
