# Implementation Plan: Rust-Native Canonical Operation Completion

## Phase 1: Operation Inventory and Red Tests [checkpoint: ed0640a]

- [x] Task: Build the operation gap inventory 2741aac
    - [x] Parse the Rust migration inventory by operation, owner, fallback status, and promotion gates
    - [x] Identify native candidates, bridge defaults, and Python-reference boundaries
    - [x] Commit this task before starting the next task
- [x] Task: Write failing operation ownership tests bff8d19
    - [x] Require every canonical operation to have an explicit native or promoted ownership state
    - [x] Require evidence gates for every native-default slice
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Operation Inventory and Red Tests' (Protocol in workflow.md)

## Phase 2: Native Operation Slice Implementation

- [ ] Task: Promote eligible `fit_model` slices
    - [ ] Implement or complete Rust-native operation paths for stable fitted-state payloads
    - [ ] Add parity and error-mapping tests
    - [ ] Commit this task before starting the next task
- [ ] Task: Promote eligible `predict_model` and `simulate_model` slices
    - [ ] Implement missing native paths where schemas are stable
    - [ ] Preserve explicit unsupported-native errors for unstable payloads
    - [ ] Commit this task before starting the next task
- [ ] Task: Promote eligible `summarize_model` and `diagnose_model` slices
    - [ ] Add native summary and diagnostics paths where deterministic payloads are stable
    - [ ] Add parity tests against Python reference responses
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Native Operation Slice Implementation' (Protocol in workflow.md)

## Phase 3: Evidence and Inventory Closure

- [ ] Task: Capture benchmark and memory evidence
    - [ ] Run Criterion or project-native Rust benchmarks for promoted slices
    - [ ] Capture DHAT or not-applicable rationale for memory-sensitive slices
    - [ ] Commit this task before starting the next task
- [ ] Task: Update inventory and roadmap evidence
    - [ ] Update `rust_core_migration_inventory.json`
    - [ ] Update Rust roadmap prose to match the machine-readable state
    - [ ] Commit this task before starting the next task
- [ ] Task: Run binding smoke matrix
    - [ ] Validate Python, Rust, R, Julia, TypeScript, Go, and C# bindings where available
    - [ ] Record blocked or not-applicable binding evidence explicitly
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Evidence and Inventory Closure' (Protocol in workflow.md)
