# Implementation Plan

## Phase 1: Advanced API Contracts [checkpoint: 49187a3]

- [x] Task: Define advanced capability contracts [5bbcbae]
    - [x] Specify stable and experimental APIs for ensembles, policy scenarios, streaming updates, and uncertainty calibration
    - [x] Add serialization and capability metadata for result objects
    - [x] Add fail-first tests for API shape and dependency fallback behavior
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add advanced workflow fixtures [43125f9]
    - [x] Create representative datasets for ensemble, policy, streaming, and calibration workflows
    - [x] Add reproducibility metadata and fixture validation tests
    - [x] Document fixture assumptions
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Advanced API Contracts' (Protocol in workflow.md)

## Phase 2: Modeling Feature Implementation [checkpoint: cb60d7d]

- [x] Task: Implement ensemble and scenario workflows [0369998]
    - [x] Add ensemble composition and scoring support
    - [x] Add causal-policy scenario comparison with auditable assumptions
    - [x] Add integration tests from fit to scenario summary
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Implement streaming updates and uncertainty calibration [a8d3e5f]
    - [x] Add incremental update support for selected stable fitted models
    - [x] Add prediction interval calibration and backtesting utilities
    - [x] Add residual and coverage diagnostics
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Modeling Feature Implementation' (Protocol in workflow.md)

## Phase 3: Accelerator Policy, Examples, and Validation [checkpoint: 552c099]

- [x] Task: Add accelerator-aware execution policy [c0af6ab]
    - [x] Implement capability-based routing for NumPy, JAX, and Rust-native paths
    - [x] Add safe fallback tests for missing optional dependencies
    - [x] Record performance evidence for selected workflows
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add advanced examples and docs [62da688]
    - [x] Add end-to-end examples for ensemble forecasts, policy scenarios, streaming updates, and calibrated intervals
    - [x] Add docs pages and Starlight routes
    - [x] Validate examples in CI or a documented release lane
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Accelerator Policy, Examples, and Validation' (Protocol in workflow.md)
