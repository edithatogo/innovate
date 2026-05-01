# Implementation Plan: Hosted Services and Remote Execution

## Phase 1: Contract and Risk Model

- [x] Task: Define remote execution boundaries
    - [x] Identify operations eligible for remote execution
    - [x] Specify request, response, error, provenance, and version fields
    - [x] Include backend provenance for NumPy/SciPy, JAX/XLA, Rust-native, and bridge fallback execution
    - [x] Reuse existing kernel schemas and Arrow-compatible interchange where possible
- [x] Task: Define security and observability expectations
    - [x] Document authentication, authorization, tenant isolation, and data retention requirements
    - [x] Define structured logging, tracing, metrics, and request correlation fields
    - [x] Document local-only operations and blocked execution patterns
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Contract and Risk Model' (Protocol in workflow.md)

## Phase 2: Prototype Slice

- [x] Task: Add contract tests for remote execution
    - [x] Write failing tests for request and response compatibility
    - [x] Write failing tests for structured remote errors
    - [x] Add observability field checks for request correlation
- [x] Task: Implement a minimal testable remote adapter
    - [x] Add a local or in-process service adapter for one eligible operation
    - [x] Preserve kernel schema versioning and provenance fields
    - [x] Avoid production infrastructure assumptions
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Prototype Slice' (Protocol in workflow.md)

## Phase 3: Documentation and CI

- [x] Task: Document remote execution usage and limits
    - [x] Explain service boundaries, eligible operations, and threat model assumptions
    - [x] Document structured errors and observability fields
    - [x] Record deployment prerequisites for future hosted work
- [x] Task: Run validation gates
    - [x] Run remote execution contract tests
    - [x] Run relevant lint, type, and documentation checks
    - [x] Confirm local API behavior remains unchanged
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and CI' (Protocol in workflow.md)
