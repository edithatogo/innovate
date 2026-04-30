# Implementation Plan: XLA Backend Strategy and JAX Kernel Promotion Gates

## Phase 1: Strategy and Eligibility Policy

- [x] Task: Define XLA eligibility rules
    - [x] Specify kernel traits that make JAX/XLA practical
    - [x] Specify rejection criteria for highly dynamic Python or event-driven workflows
    - [x] Define deterministic PRNG and shape-stability expectations
- [x] Task: Define preferred XLA library roles
    - [x] Document JAX, NumPyro, BlackJAX, TensorFlow Probability JAX substrate, and Diffrax roles
    - [x] Document when NumPy/SciPy remains the reference path
    - [x] Document optional dependency and backend-gate requirements
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Strategy and Eligibility Policy' (Protocol in workflow.md)

## Phase 2: Promotion Gates and Validation

- [x] Task: Add XLA promotion gate tests
    - [x] Write checks that roadmap and active backlog tracks reference XLA evaluation gates
    - [x] Write checks that base-install documentation keeps JAX optional
    - [x] Write checks that compile-time and steady-state benchmark reporting are required
- [x] Task: Implement documentation and governance updates
    - [x] Update roadmap and architecture docs with XLA-first optional-backend guidance
    - [x] Update backlog track specs and plans with JAX/XLA evaluation tasks
    - [x] Cross-link XLA policy from benchmark and probabilistic follow-on work
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Promotion Gates and Validation' (Protocol in workflow.md)

## Phase 3: Track Integration

- [x] Task: Integrate XLA strategy with future implementation tracks
    - [x] Ensure probabilistic inference prefers NumPyro and BlackJAX for eligible kernels
    - [x] Ensure diagnostics and uncertainty work evaluates JAX-compatible artifact generation
    - [x] Ensure benchmark automation separates JIT compile cost from steady-state runtime
    - [x] Ensure operational simulation and DataFrame experiments document XLA eligibility or rejection
    - [x] Ensure Rust-core promotion compares native Rust, Python reference, and XLA-backed paths where applicable
- [x] Task: Run validation gates
    - [x] Run roadmap mapping and XLA strategy tests
    - [x] Run relevant lint, format, and documentation checks
    - [x] Confirm the worktree has no unintended changes
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Track Integration' (Protocol in workflow.md)
