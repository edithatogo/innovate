# Implementation Plan: Probabilistic Inference Expansion

## Phase 1: Scope and Contract

- [x] Task: Inventory current probabilistic inference surfaces
    - [x] Identify existing Bayesian, stochastic, and simulation-backed entry points
    - [x] Map each entry point to current dependency and schema behavior
    - [x] Record gaps between existing behavior and the functional kernel contract
- [x] Task: Define candidate expansion slices
    - [x] Select model families or inference routines for the first expansion slice
    - [x] Evaluate XLA eligibility and prefer NumPyro or BlackJAX where kernel shape and PRNG requirements fit
    - [x] Document any reason for rejecting XLA-backed inference before selecting a non-XLA engine
    - [x] Define posterior, uncertainty, diagnostics, and provenance payload requirements
    - [x] Document promotion criteria for experimental and supported tiers
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Scope and Contract' (Protocol in workflow.md)

## Phase 2: First Implementation Slice

- [x] Task: Add schema and fixture tests for the selected slice
    - [x] Write failing tests for request and response payload compatibility
    - [x] Add deterministic stochastic fixtures with fixed seeds or stable summaries
    - [x] Verify optional dependency failures produce structured errors
- [x] Task: Implement the selected probabilistic inference slice
    - [x] Add the minimal runtime integration behind optional backend gates
    - [x] Separate JIT compilation cost from steady-state inference benchmarks where JAX/XLA is used
    - [x] Preserve deterministic baseline semantics where applicable
    - [x] Emit versioned uncertainty, posterior, diagnostics, and provenance artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: First Implementation Slice' (Protocol in workflow.md)

## Phase 3: Documentation and Validation

- [x] Task: Document probabilistic inference support
    - [x] Update user docs with supported routines, dependencies, and promotion status
    - [x] Document output schemas and stochastic reproducibility expectations
    - [x] Add release notes or roadmap updates describing the new coverage
- [x] Task: Run validation gates
    - [x] Run focused tests for probabilistic schema and backend behavior
    - [x] Run relevant lint, type, and documentation checks
    - [x] Confirm base install imports still work without optional inference engines
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Validation' (Protocol in workflow.md)
