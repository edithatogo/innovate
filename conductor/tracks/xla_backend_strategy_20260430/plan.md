# Implementation Plan: XLA Backend Strategy and JAX Kernel Promotion Gates

## Phase 1: Strategy and Eligibility Policy

- [ ] Task: Define XLA eligibility rules
    - [ ] Specify kernel traits that make JAX/XLA practical
    - [ ] Specify rejection criteria for highly dynamic Python or event-driven workflows
    - [ ] Define deterministic PRNG and shape-stability expectations
- [ ] Task: Define preferred XLA library roles
    - [ ] Document JAX, NumPyro, BlackJAX, TensorFlow Probability JAX substrate, and Diffrax roles
    - [ ] Document when NumPy/SciPy remains the reference path
    - [ ] Document optional dependency and backend-gate requirements
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Strategy and Eligibility Policy' (Protocol in workflow.md)

## Phase 2: Promotion Gates and Validation

- [ ] Task: Add XLA promotion gate tests
    - [ ] Write checks that roadmap and active backlog tracks reference XLA evaluation gates
    - [ ] Write checks that base-install documentation keeps JAX optional
    - [ ] Write checks that compile-time and steady-state benchmark reporting are required
- [ ] Task: Implement documentation and governance updates
    - [ ] Update roadmap and architecture docs with XLA-first optional-backend guidance
    - [ ] Update backlog track specs and plans with JAX/XLA evaluation tasks
    - [ ] Cross-link XLA policy from benchmark and probabilistic follow-on work
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Promotion Gates and Validation' (Protocol in workflow.md)

## Phase 3: Track Integration

- [ ] Task: Integrate XLA strategy with future implementation tracks
    - [ ] Ensure probabilistic inference prefers NumPyro and BlackJAX for eligible kernels
    - [ ] Ensure diagnostics and uncertainty work evaluates JAX-compatible artifact generation
    - [ ] Ensure benchmark automation separates JIT compile cost from steady-state runtime
    - [ ] Ensure operational simulation and DataFrame experiments document XLA eligibility or rejection
    - [ ] Ensure Rust-core promotion compares native Rust, Python reference, and XLA-backed paths where applicable
- [ ] Task: Run validation gates
    - [ ] Run roadmap mapping and XLA strategy tests
    - [ ] Run relevant lint, format, and documentation checks
    - [ ] Confirm the worktree has no unintended changes
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Track Integration' (Protocol in workflow.md)
