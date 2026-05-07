# Implementation Plan

## Phase 1: Build migration backlog

- [x] Task: Enumerate remaining Rust migration slices
    - [x] Model families
    - [x] Canonical kernel operations
    - [x] Unsupported payload shapes
    - [x] Python-only reference paths
- [x] Task: Conductor - Automated Review and Checkpoint 'Build migration backlog' (Protocol in workflow.md)

## Phase 2: Define promotion gates

- [x] Task: Add operation-level promotion checklists
    - [x] Parity
    - [x] Schema compatibility
    - [x] Error mapping
    - [x] Benchmark and memory evidence
    - [x] Binding smoke tests
- [x] Task: Conductor - Automated Review and Checkpoint 'Define promotion gates' (Protocol in workflow.md)

## Phase 3: Validate execution plan

- [x] Task: Add migration governance tests
    - [x] Ensure every slice has a state
    - [x] Ensure dependencies are explicit
    - [x] Ensure no unsupported default is claimed
- [x] Task: Conductor - Automated Review and Checkpoint 'Validate execution plan' (Protocol in workflow.md)
