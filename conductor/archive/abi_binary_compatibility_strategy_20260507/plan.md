# Implementation Plan: ABI and Binary Compatibility Strategy

## Phase 1: Define Compatibility Boundaries

- [x] Task: Add policy guard tests
    - [x] Ensure API stability language exists
    - [x] Ensure XLA internals are rejected as public ABI
    - [x] Ensure native details are capability-gated
- [x] Task: Separate API, schema, and ABI contracts
    - [x] Public API
    - [x] Kernel schema
    - [x] Arrow/native ABI
    - [x] Backend capability metadata
- [x] Task: Conductor - Automated Review and Checkpoint 'Define Compatibility Boundaries' (Protocol in workflow.md)

## Phase 2: Add ABI Policy Docs

- [x] Task: Document native and accelerator boundaries
    - [x] Arrow C Data Interface
    - [x] Rust native internals
    - [x] XLA internals
    - [x] Package-manager binary compatibility
- [x] Task: Conductor - Automated Review and Checkpoint 'Add ABI Policy Docs' (Protocol in workflow.md)

## Phase 3: Validate Non-Breaking Migration

- [x] Task: Validate policy and archive track
    - [x] Run focused ABI policy tests
    - [x] Run registry/archive drift tests
    - [x] Archive the completed Conductor track
- [x] Task: Conductor - Automated Review and Checkpoint 'Validate Non-Breaking Migration' (Protocol in workflow.md)
