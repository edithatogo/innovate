# Implementation Plan

## Phase 1: Build migration backlog

- [ ] Task: Enumerate remaining Rust migration slices
    - [ ] Model families
    - [ ] Canonical kernel operations
    - [ ] Unsupported payload shapes
    - [ ] Python-only reference paths
- [ ] Task: Conductor - Automated Review and Checkpoint 'Build migration backlog' (Protocol in workflow.md)

## Phase 2: Define promotion gates

- [ ] Task: Add operation-level promotion checklists
    - [ ] Parity
    - [ ] Schema compatibility
    - [ ] Error mapping
    - [ ] Benchmark and memory evidence
    - [ ] Binding smoke tests
- [ ] Task: Conductor - Automated Review and Checkpoint 'Define promotion gates' (Protocol in workflow.md)

## Phase 3: Validate execution plan

- [ ] Task: Add migration governance tests
    - [ ] Ensure every slice has a state
    - [ ] Ensure dependencies are explicit
    - [ ] Ensure no unsupported default is claimed
- [ ] Task: Conductor - Automated Review and Checkpoint 'Validate execution plan' (Protocol in workflow.md)
