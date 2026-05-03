# Implementation Plan: MARS Surrogate Benchmark Gate

## Phase 1: Benchmark Design

- [ ] Task: Define benchmark candidates and baselines
    - [ ] Identify adoption-curve surrogate scenarios and expected outputs
    - [ ] Compare against NumPy/SciPy reference behavior and eligible JAX/XLA-backed alternatives
    - [ ] Define correctness tolerances, runtime tiers, and promotion thresholds
- [ ] Task: Add benchmark-governance tests
    - [ ] Write failing checks that `mars` remains optional until evidence is recorded
    - [ ] Check that benchmark metadata separates surrogate gains from XLA-backed kernel gains
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Benchmark Design' (Protocol in workflow.md)

## Phase 2: Benchmark Evidence

- [ ] Task: Add opt-in benchmark harness or fixtures
    - [ ] Add compact benchmark metadata for candidate surrogate workflows
    - [ ] Record dependency cost, failure modes, and baseline comparisons
    - [ ] Keep benchmark execution outside mandatory fast unit tests
- [ ] Task: Validate benchmark metadata
    - [ ] Run focused benchmark-governance tests
    - [ ] Run relevant benchmark dry-run or metadata checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Benchmark Evidence' (Protocol in workflow.md)

## Phase 3: Adapter Decision

- [ ] Task: Document promote, defer, or reject outcome
    - [ ] Explain whether `mars` should become an optional extra or remain conceptual
    - [ ] Record how the decision interacts with XLA-backed alternatives
    - [ ] Update ecosystem documentation and release notes as needed
- [ ] Task: Run validation gates
    - [ ] Run focused roadmap and ecosystem tests
    - [ ] Run relevant docs or prose checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Adapter Decision' (Protocol in workflow.md)
