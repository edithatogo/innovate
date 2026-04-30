# Implementation Plan: Benchmark Corpus Automation

## Phase 1: Corpus Metadata Contract

- [ ] Task: Inventory existing benchmark assets
    - [ ] List benchmark fixtures, model cards, benchmark scripts, and profiling outputs
    - [ ] Identify missing dataset metadata and freshness signals
    - [ ] Classify checks as fast CI, scheduled, or manual
- [ ] Task: Define benchmark corpus metadata
    - [ ] Specify required metadata fields for datasets and benchmark cases
    - [ ] Add fields for XLA compilation cost, steady-state runtime, accelerator target, and reference baseline
    - [ ] Define model-card freshness rules and output provenance fields
    - [ ] Document performance gate links for optional backends and Rust core promotion
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Corpus Metadata Contract' (Protocol in workflow.md)

## Phase 2: Automation

- [ ] Task: Add failing checks for benchmark corpus metadata
    - [ ] Test that every benchmark fixture has required metadata
    - [ ] Test that representative model-card outputs can be refreshed or validated
    - [ ] Test that expensive benchmarks are excluded from default fast CI
- [ ] Task: Implement benchmark automation
    - [ ] Add validation commands for benchmark metadata and model-card freshness
    - [ ] Ensure JAX/XLA benchmark reports separate first-call compilation from repeated execution
    - [ ] Wire fast checks into CI and document scheduled benchmark commands
    - [ ] Produce actionable diagnostics for stale or missing benchmark artifacts
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Automation' (Protocol in workflow.md)

## Phase 3: Documentation and Release Gates

- [ ] Task: Document benchmark contribution workflow
    - [ ] Explain how to add fixtures, metadata, and model-card summaries
    - [ ] Document when scheduled benchmarks should run
    - [ ] Cross-link performance promotion gates from Rust and backend docs
- [ ] Task: Run validation gates
    - [ ] Run benchmark metadata checks
    - [ ] Run focused unit tests for benchmark automation
    - [ ] Run relevant lint and documentation checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Release Gates' (Protocol in workflow.md)
