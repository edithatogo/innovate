# Implementation Plan: Rust Core Expansion

## Phase 1: Candidate Selection

- [ ] Task: Inventory Rust core status
    - [ ] List native Rust operations, Python bridge fallbacks, and Python-only operations
    - [ ] Map each operation to schema readiness and test coverage
    - [ ] Identify error mapping and fallback gaps
- [ ] Task: Select the next Rust-core slice
    - [ ] Choose candidate operations based on stability and benchmark value
    - [ ] Compare Rust-native suitability against eligible JAX/XLA-backed implementations
    - [ ] Define parity, error mapping, fallback, and benchmark gates
    - [ ] Document promotion criteria before implementation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Candidate Selection' (Protocol in workflow.md)

## Phase 2: Native Implementation Slice

- [ ] Task: Add parity and benchmark tests
    - [ ] Write failing parity tests against Python reference semantics
    - [ ] Write failing tests for structured error mapping and bridge fallback
    - [ ] Add focused benchmark or profiling checks for the selected operations
- [ ] Task: Implement the selected Rust-core path
    - [ ] Add native Rust logic behind existing schema boundaries
    - [ ] Preserve bridge fallback behavior where native support is incomplete
    - [ ] Expose support status without changing the public API
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Native Implementation Slice' (Protocol in workflow.md)

## Phase 3: Promotion Documentation

- [ ] Task: Document Rust-core support status
    - [ ] Update Rust core roadmap and binding docs with native, bridged, and Python-backed operations
    - [ ] Record whether XLA-backed execution was rejected, complementary, or outperformed by Rust-native execution
    - [ ] Record benchmark evidence and promotion decision
    - [ ] Document fallback and error behavior for binding consumers
- [ ] Task: Run validation gates
    - [ ] Run Rust, Python, and binding schema compatibility checks for the selected slice
    - [ ] Run benchmark or profiling checks required by the promotion criteria
    - [ ] Run relevant lint, type, and documentation checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Promotion Documentation' (Protocol in workflow.md)
