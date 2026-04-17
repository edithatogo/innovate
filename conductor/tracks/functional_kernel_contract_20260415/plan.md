# Implementation Plan: Functional Kernel Contract

## Phase 1: Contract Definition

- [x] Task: Define the kernel surface [01c7feb]
    - [x] Specify canonical kernel operations for discovery, fit, predict, simulate, and diagnostics
    - [x] Define versioned request and response schemas
    - [x] Ensure the contract is Array API-friendly at the numerical boundary
    - [x] Write failing contract-validation tests
- [x] Task: Define serialization and error semantics [01c7feb]
    - [x] Specify canonical scalar, array, metadata, and Arrow-compatible tabular encoding rules
    - [x] Define stable error codes and payloads
    - [x] Confirm the contract tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Contract Definition' (Protocol in workflow.md) [01c7feb]

## Phase 2: Core Model Adapters [checkpoint: e3ec496]

- [x] Task: Implement kernel adapters for stable models
    - [x] Add adapters for the stable diffusion, substitution, and competition families
    - [x] Ensure adapter outputs match the documented schemas
    - [x] Make the adapter tests pass
- [x] Task: Implement model discovery and capability exposure
    - [x] Wire the kernel to the canonical capability registry
    - [x] Return machine-readable model metadata
    - [x] Verify deterministic behavior for stable model discovery
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Model Adapters' (Protocol in workflow.md)

## Phase 3: Versioning and Documentation

- [~] Task: Add schema versioning and compatibility checks
    - [ ] Implement version markers and compatibility validation
    - [ ] Add tests for backward-compatible request handling where supported
    - [ ] Document migration expectations for future revisions
- [ ] Task: Document kernel usage and boundaries
    - [ ] Add guidance for binding authors and advanced users
    - [ ] Explain how the kernel complements the OO Python API and Arrow interchange layer
    - [ ] Verify all acceptance criteria are satisfied
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Versioning and Documentation' (Protocol in workflow.md)
