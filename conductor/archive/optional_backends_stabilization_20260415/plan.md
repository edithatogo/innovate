# Implementation Plan: Optional Dependency Isolation, Backend Stabilization, and Engine Positioning

## Phase 1: Base vs Optional Capability Audit [checkpoint: ebc5af2]

- [x] Task: Audit base-import and test-collection behavior
    - [x] Identify modules that import optional dependencies at import time
    - [x] Identify tests that should be base-only versus optional-backend coverage
    - [x] Define the supported installation modes for contributors and users
    - [x] Record the intended support boundaries for pandas, PyArrow, and any selective Polars usage
- [x] Task: Write failing coverage for dependency isolation
    - [x] Add tests covering base import behavior without optional extras
    - [x] Add tests for backend capability discovery
    - [x] Add tests for missing-optional-dependency error messaging
- [x] Task: Conductor - User Manual Verification 'Phase 1: Base vs Optional Capability Audit' (Protocol in workflow.md)

## Phase 2: Implement Isolation and Capability Metadata [checkpoint: ebc5af2]

- [x] Task: Isolate optional dependency imports
    - [x] Refactor optional backends to lazy-import or guarded-import patterns
    - [x] Prevent optional scientific stacks from being required by default imports
    - [x] Ensure unstable Bayesian paths are clearly marked experimental
    - [x] Ensure JAX/XLA implementation details do not leak into the durable public surface
- [x] Task: Implement backend capability metadata
    - [x] Define a canonical representation of backend/model support
    - [x] Expose capability inspection to callers
    - [x] Align capability metadata with current implementation reality
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implement Isolation and Capability Metadata' (Protocol in workflow.md)

## Phase 3: Test Matrix and Documentation [checkpoint: ebc5af2]

- [x] Task: Separate the test strategy for base and optional extras
    - [x] Define base and optional test selectors
    - [x] Ensure local and CI commands reflect the split
    - [x] Validate acceptance criteria in both environments
- [x] Task: Document supported backend modes
    - [x] Update installation guidance for extras
    - [x] Document backend limitations and support levels
    - [x] Add troubleshooting guidance for missing optional dependencies
    - [x] Document the pandas plus PyArrow default and the selective-use policy for Polars
- [x] Task: Conductor - User Manual Verification 'Phase 3: Test Matrix and Documentation' (Protocol in workflow.md)
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Test Matrix and Documentation' (Protocol in workflow.md) [ebc5af2]
