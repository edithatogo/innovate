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

## Phase 2: Native Operation Slice Implementation [checkpoint: 892a181]

- [x] Task: Promote eligible `fit_model` slices 8116b0a
    - [x] Implement or complete Rust-native operation paths for stable fitted-state payloads
    - [x] Add parity and error-mapping tests
    - [x] Commit this task before starting the next task
- [x] Task: Promote eligible `predict_model` and `simulate_model` slices 5d6f91a
    - [x] Implement missing native paths where schemas are stable
    - [x] Preserve explicit unsupported-native errors for unstable payloads
    - [x] Commit this task before starting the next task
- [x] Task: Promote eligible `summarize_model` and `diagnose_model` slices 7039871
    - [x] Add native summary and diagnostics paths where deterministic payloads are stable
    - [x] Add parity tests against Python reference responses
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Native Operation Slice Implementation' (Protocol in workflow.md)

## Phase 3: Evidence and Inventory Closure [checkpoint: pending]

- [x] Task: Capture benchmark and memory evidence e3daea5
    - [x] Run Criterion or project-native Rust benchmarks for promoted slices
    - [x] Capture DHAT or not-applicable rationale for memory-sensitive slices
    - [x] Commit this task before starting the next task
- [x] Task: Update inventory and roadmap evidence 17c795f
    - [x] Update `rust_core_migration_inventory.json`
    - [x] Update Rust roadmap prose to match the machine-readable state
    - [x] Commit this task before starting the next task
- [x] Task: Run binding smoke matrix 6835542
    - [x] Validate Python, Rust, R, Julia, TypeScript, Go, and C# bindings where available
    - [x] Record blocked or not-applicable binding evidence explicitly
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Evidence and Inventory Closure' (Protocol in workflow.md)
