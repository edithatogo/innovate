# Implementation Plan: DataFrame Engine Experimentation

## Phase 1: Workload Selection

- [ ] Task: Inventory current tabular engine usage
    - [ ] Map pandas, PyArrow, and Polars usage across the repository
    - [ ] Identify workloads with measurable ETL, ingestion, or benchmark pressure
    - [ ] Document public API surfaces that must remain engine-neutral
- [ ] Task: Define experiment criteria
    - [ ] Select candidate workloads for optional engine experimentation
    - [ ] Define correctness and performance metrics
    - [ ] Separate DataFrame-engine effects from XLA-backed numerical-kernel effects
    - [ ] Document dependency, fallback, and support-tier boundaries
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Workload Selection' (Protocol in workflow.md)

## Phase 2: Experiment Implementation

- [ ] Task: Add correctness and benchmark fixtures
    - [ ] Write failing correctness tests for selected engine-neutral behavior
    - [ ] Add benchmark fixtures comparing pandas plus PyArrow and optional engine paths
    - [ ] Add dependency-gate tests for missing optional engines
- [ ] Task: Implement selected optional engine path
    - [ ] Add the minimal engine-specific implementation behind explicit gates
    - [ ] Preserve public API inputs, outputs, and schema payloads
    - [ ] Record benchmark results and fallback behavior
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Experiment Implementation' (Protocol in workflow.md)

## Phase 3: Decision and Documentation

- [ ] Task: Document experiment results
    - [ ] Summarize correctness, performance, and memory tradeoffs
    - [ ] Attribute wins to tabular execution, XLA-backed kernels, or their interaction
    - [ ] Document whether the path is supported, experimental, or rejected
    - [ ] Update architecture docs if engine strategy changes
- [ ] Task: Run validation gates
    - [ ] Run tabular engine tests and focused benchmarks
    - [ ] Run relevant lint, type, and documentation checks
    - [ ] Confirm public API behavior is engine-neutral
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Decision and Documentation' (Protocol in workflow.md)
