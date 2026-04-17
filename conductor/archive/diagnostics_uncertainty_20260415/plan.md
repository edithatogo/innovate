# Implementation Plan: Standard Diagnostics, Uncertainty, and Model Comparison

## Phase 1: Diagnostics Contract Definition

- [x] Task: Define the common diagnostics and uncertainty surface
    - [x] Inventory current metric, residual, and uncertainty outputs
    - [x] Define the canonical result structure for diagnostics and warnings
    - [x] Define comparison semantics across stable model families
- [x] Task: Write failing contract tests
    - [x] Add tests for diagnostics result shape
    - [x] Add tests for supported uncertainty output variants
    - [x] Add tests for model comparison behavior and unsupported cases
- [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnostics Contract Definition' (Protocol in workflow.md)

## Phase 2: Implement Diagnostics Standardization

- [x] Task: Implement the shared diagnostics layer
    - [x] Normalize fit-quality metric reporting
    - [x] Normalize residual-analysis outputs
    - [x] Normalize warnings and support-level metadata
- [x] Task: Integrate stable model families
    - [x] Update deterministic fitters and models to emit the shared diagnostics surface
    - [x] Update probabilistic or bootstrap-capable paths to map onto the same contract
    - [x] Ensure unsupported diagnostics are explicit rather than implicit
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implement Diagnostics Standardization' (Protocol in workflow.md)

## Phase 3: Comparison Tools and Documentation

- [x] Task: Update comparison and plotting utilities
    - [x] Align comparison helpers with the shared diagnostics contract
    - [x] Align visual diagnostics with the programmatic outputs
    - [x] Verify stable behavior across representative model families
- [x] Task: Document the diagnostics layer
    - [x] Add API docs and usage examples
    - [x] Document uncertainty provenance and interpretation
    - [x] Add acceptance-criteria coverage to tests and docs
- [x] Task: Conductor - User Manual Verification 'Phase 3: Comparison Tools and Documentation' (Protocol in workflow.md)
