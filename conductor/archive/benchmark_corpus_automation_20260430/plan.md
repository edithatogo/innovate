# Implementation Plan: Benchmark Corpus Automation

## Phase 1: Corpus Metadata Contract

- [x] Task: Inventory existing benchmark assets
    - [x] List benchmark fixtures, model cards, benchmark scripts, and profiling outputs
    - [x] Identify missing dataset metadata and freshness signals
    - [x] Classify checks as fast CI, scheduled, or manual
- [x] Task: Define benchmark corpus metadata
    - [x] Specify required metadata fields for datasets and benchmark cases
    - [x] Add fields for XLA compilation cost, steady-state runtime, accelerator target, and reference baseline
    - [x] Define model-card freshness rules and output provenance fields
    - [x] Document performance gate links for optional backends and Rust core promotion
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Corpus Metadata Contract' (Protocol in workflow.md)

## Phase 2: Automation

- [x] Task: Add failing checks for benchmark corpus metadata
    - [x] Test that every benchmark fixture has required metadata
    - [x] Test that representative model-card outputs can be refreshed or validated
    - [x] Test that expensive benchmarks are excluded from default fast CI
- [x] Task: Implement benchmark automation
    - [x] Add validation commands for benchmark metadata and model-card freshness
    - [x] Ensure JAX/XLA benchmark reports separate first-call compilation from repeated execution
    - [x] Wire fast checks into CI and document scheduled benchmark commands
    - [x] Produce actionable diagnostics for stale or missing benchmark artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Automation' (Protocol in workflow.md)

## Phase 3: Documentation and Release Gates

- [x] Task: Document benchmark contribution workflow
    - [x] Explain how to add fixtures, metadata, and model-card summaries
    - [x] Document when scheduled benchmarks should run
    - [x] Cross-link performance promotion gates from Rust and backend docs
- [x] Task: Run validation gates
    - [x] Run benchmark metadata checks
    - [x] Run focused unit tests for benchmark automation
    - [x] Run relevant lint and documentation checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Release Gates' (Protocol in workflow.md)
