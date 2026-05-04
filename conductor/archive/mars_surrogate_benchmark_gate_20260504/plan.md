# Implementation Plan: MARS Surrogate Benchmark Gate

## Phase 1: Benchmark Design

- [x] Task: Define benchmark candidates and baselines
    - [x] Identify adoption-curve surrogate scenarios and expected outputs
    - [x] Compare against NumPy/SciPy reference behavior and eligible JAX/XLA-backed alternatives
    - [x] Define correctness tolerances, runtime tiers, and promotion thresholds
- [x] Task: Add benchmark-governance tests
    - [x] Write failing checks that `mars` remains optional until evidence is recorded
    - [x] Check that benchmark metadata separates surrogate gains from XLA-backed kernel gains
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Benchmark Design' (Protocol in workflow.md)

## Phase 2: Benchmark Evidence

- [x] Task: Add opt-in benchmark harness or fixtures
    - [x] Add compact benchmark metadata for candidate surrogate workflows
    - [x] Record dependency cost, failure modes, and baseline comparisons
    - [x] Keep benchmark execution outside mandatory fast unit tests
- [x] Task: Validate benchmark metadata
    - [x] Run focused benchmark-governance tests
    - [x] Run relevant benchmark dry-run or metadata checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Benchmark Evidence' (Protocol in workflow.md)

## Phase 3: Adapter Decision

- [x] Task: Document promote, defer, or reject outcome
    - [x] Explain whether `mars` should become an optional extra or remain conceptual
    - [x] Record how the decision interacts with XLA-backed alternatives
    - [x] Update ecosystem documentation and release notes as needed
- [x] Task: Run validation gates
    - [x] Run focused roadmap and ecosystem tests
    - [x] Run relevant docs or prose checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Adapter Decision' (Protocol in workflow.md)
