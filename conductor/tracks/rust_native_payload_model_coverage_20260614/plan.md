# Implementation Plan: Rust-Native Payload and Model-Family Coverage

## Phase 1: Registry and Payload Red Tests [checkpoint: 1ba624c]

- [x] Task: Inventory Python registry model families 43b33ef
    - [x] Compare Python capability registry entries with Rust-native model slices
    - [x] Identify composite, multi-product, network, policy, probabilistic, and ecosystem families requiring classification
    - [x] Commit this task before starting the next task
- [x] Task: Inventory stable payload shapes 7c8ea6e
    - [x] List fitted-state, covariate, event, diagnostics, simulation, and uncertainty payload shapes
    - [x] Mark each payload as stable, provisional, internal, or Python-reference-only
    - [x] Commit this task before starting the next task
- [x] Task: Write failing classification tests 797779c
    - [x] Require every model family to have ownership status
    - [x] Require every stable payload shape to have schema and ownership evidence
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Registry and Payload Red Tests' (Protocol in workflow.md)

## Phase 2: Model-Family Coverage Slices

- [x] Task: Promote stable diffusion and substitution family payloads aa8c7bc
    - [x] Add Rust-native support or explicit non-native promotion for remaining stable slices
    - [x] Add parity fixtures and error mapping tests
    - [x] Commit this task before starting the next task
- [ ] Task: Classify composite and multi-product families
    - [ ] Implement native support where schemas are stable
    - [ ] Keep bridge fallback only with explicit rationale and tests
    - [ ] Commit this task before starting the next task
- [ ] Task: Classify network, policy, and ecosystem families
    - [ ] Promote only stable deterministic payloads
    - [ ] Keep object-internal or agent-based behavior Python-reference-owned until schema boundaries exist
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Model-Family Coverage Slices' (Protocol in workflow.md)

## Phase 3: Full-Claim Gate

- [ ] Task: Update full Rust ownership gate
    - [ ] Add a machine-readable gate that determines whether full Rust ownership may be claimed
    - [ ] Ensure docs consume the gate rather than hand-written claims
    - [ ] Commit this task before starting the next task
- [ ] Task: Run full Rust ownership validation
    - [ ] Run Rust, Python, and binding tests for promoted families
    - [ ] Record intentionally excluded Python-reference boundaries
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Full-Claim Gate' (Protocol in workflow.md)
