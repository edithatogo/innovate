# Implementation Plan: DataFrame Engine Experimentation

## Phase 1: Workload Selection

- [x] Task: Inventory current tabular engine usage
    - [x] Map pandas, PyArrow, and Polars usage across the repository
    - [x] Identify workloads with measurable ETL, ingestion, or benchmark pressure
    - [x] Document public API surfaces that must remain engine-neutral
- [x] Task: Define experiment criteria
    - [x] Select candidate workloads for optional engine experimentation
    - [x] Define correctness and performance metrics
    - [x] Separate DataFrame-engine effects from XLA-backed numerical-kernel effects
    - [x] Document dependency, fallback, and support-tier boundaries
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Workload Selection' (Protocol in workflow.md)

## Phase 2: Experiment Implementation

- [x] Task: Add correctness and benchmark fixtures
    - [x] Write failing correctness tests for selected engine-neutral behavior
    - [x] Add benchmark fixtures comparing pandas plus PyArrow and optional engine paths
    - [x] Add dependency-gate tests for missing optional engines
- [x] Task: Implement selected optional engine path
    - [x] Add the minimal engine-specific implementation behind explicit gates
    - [x] Preserve public API inputs, outputs, and schema payloads
    - [x] Record benchmark results and fallback behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Experiment Implementation' (Protocol in workflow.md)

## Phase 3: Decision and Documentation

- [x] Task: Document experiment results
    - [x] Summarize correctness, performance, and memory tradeoffs
    - [x] Attribute wins to tabular execution, XLA-backed kernels, or their interaction
    - [x] Document whether the path is supported, experimental, or rejected
    - [x] Update architecture docs if engine strategy changes
- [x] Task: Run validation gates
    - [x] Run tabular engine tests and focused benchmarks
    - [x] Run relevant lint, type, and documentation checks
    - [x] Confirm public API behavior is engine-neutral
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Decision and Documentation' (Protocol in workflow.md)
