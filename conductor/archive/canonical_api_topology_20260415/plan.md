# Implementation Plan: Canonical Public API and Package Topology

## Phase 1: Inventory and Target Topology

- [x] Task: Inventory the current public API surface
    - [x] Review imports used by docs, examples, and tests
    - [x] Identify duplicate or conflicting namespaces and symbols
    - [x] Record the current public modules that should remain stable
- [x] Task: Define the target package topology
    - [x] Specify the canonical import path for each major subsystem
    - [x] Draft a compatibility map for legacy or duplicate paths
    - [x] Define the shape of the model capability registry
- [x] Task: Write failing API contract tests
    - [x] Add tests for canonical top-level imports
    - [x] Add tests for deprecated import-path behavior
    - [x] Add tests for capability registry discovery
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Inventory and Target Topology' (Protocol in workflow.md)

## Phase 2: Implement the Canonical API

- [x] Task: Implement minimal package changes to satisfy the API contract
    - [x] Update `__init__` exports and package module boundaries
    - [x] Introduce compatibility shims or warnings for deprecated paths
    - [x] Remove or isolate duplicate namespace entry points where appropriate
- [x] Task: Implement the capability registry
    - [x] Add registry definitions for stable model families
    - [x] Expose registry access from a canonical package location
    - [x] Ensure registry metadata is consistent with actual implementation support
- [x] Task: Verify canonical import behavior
    - [x] Run the new API tests and fix failures
    - [x] Validate examples and docs import paths against the canonical topology
    - [x] Confirm no circular-import regressions were introduced
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implement the Canonical API' (Protocol in workflow.md)

## Phase 3: Documentation and Compatibility Cleanup

- [x] Task: Update user-facing documentation for the canonical API
    - [x] Refresh README and docs examples to use canonical imports
    - [x] Add a public API reference section for stable imports
    - [x] Document the capability registry and compatibility policy
- [x] Task: Finalize compatibility guidance
    - [x] Document migration guidance for deprecated import paths
    - [x] Add release-note entries for the topology change
    - [x] Verify all acceptance criteria are satisfied
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation and Compatibility Cleanup' (Protocol in workflow.md)
