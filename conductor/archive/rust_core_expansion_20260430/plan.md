# Implementation Plan: Rust Core Expansion

## Phase 1: Candidate Selection

- [x] Task: Inventory Rust core status
    - [x] List native Rust operations, Python bridge fallbacks, and Python-only operations
    - [x] Map each operation to schema readiness and test coverage
    - [x] Identify error mapping and fallback gaps
- [x] Task: Select the next Rust-core slice
    - [x] Choose candidate operations based on stability and benchmark value
    - [x] Compare Rust-native suitability against eligible JAX/XLA-backed implementations
    - [x] Define parity, error mapping, fallback, and benchmark gates
    - [x] Document promotion criteria before implementation
    - [x] Record the operation support inventory with native Rust scope, bridge fallback scope, Python-only reference scope, and Rust vs JAX/XLA eligibility
    - [x] Require a benchmark promotion dossier before default promotion
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Candidate Selection' (Protocol in workflow.md)

## Phase 2: Native Implementation Slice

- [x] Task: Add parity and benchmark tests
    - [x] Write failing parity tests against Python reference semantics
    - [x] Write failing tests for structured error mapping and bridge fallback
    - [x] Add focused benchmark or profiling checks for the selected operations
- [x] Task: Implement the selected Rust-core path
    - [x] Add native Rust logic behind existing schema boundaries
    - [x] Preserve bridge fallback behavior where native support is incomplete
    - [x] Expose support status without changing the public API
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Native Implementation Slice' (Protocol in workflow.md)

## Phase 3: Promotion Documentation

- [x] Task: Document Rust-core support status
    - [x] Update Rust core roadmap and binding docs with native, bridged, and Python-backed operations
    - [x] Record whether XLA-backed execution was rejected, complementary, or outperformed by Rust-native execution
    - [x] Record benchmark evidence and promotion decision
    - [x] Document fallback and error behavior for binding consumers
- [x] Task: Run validation gates
    - [x] Run Rust, Python, and binding schema compatibility checks for the selected slice
    - [x] Run benchmark or profiling checks required by the promotion criteria
    - [x] Run relevant lint, type, and documentation checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Promotion Documentation' (Protocol in workflow.md)
