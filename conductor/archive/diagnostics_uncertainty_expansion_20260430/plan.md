# Implementation Plan: Diagnostics and Uncertainty Expansion

## Phase 1: Artifact Contract

- [x] Task: Inventory diagnostics and uncertainty outputs
    - [x] List existing diagnostic functions, summaries, and model-comparison helpers
    - [x] Map outputs to schema and Arrow interchange support
    - [x] Identify diagnostics currently tied to private Python objects
- [x] Task: Define the richer diagnostics contract
    - [x] Specify artifact payloads for residuals, calibration, uncertainty, and model comparison
    - [x] Evaluate JAX/XLA eligibility for array-heavy diagnostics and uncertainty summaries
    - [x] Define support tiers and promotion criteria by model family
    - [x] Document compatibility expectations for bindings
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Artifact Contract' (Protocol in workflow.md)

## Phase 2: Implementation Slice

- [x] Task: Add diagnostics contract tests
    - [x] Write failing tests for selected diagnostic artifact schemas
    - [x] Add representative cross-language fixture payloads
    - [x] Add tolerance-based checks for stochastic or approximate outputs
- [x] Task: Implement selected richer diagnostics
    - [x] Add the minimal diagnostics logic for the chosen model family or workflow
    - [x] Use a JAX/XLA-backed path for eligible kernels or document why the selected diagnostics are not XLA-suitable
    - [x] Emit stable schema-compatible artifact payloads
    - [x] Preserve current public API compatibility
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Implementation Slice' (Protocol in workflow.md)

## Phase 3: Documentation and Gates

- [x] Task: Update diagnostics documentation
    - [x] Document supported artifacts, model families, and interpretation notes
    - [x] Add examples for consuming diagnostics through kernel payloads
    - [x] Update roadmap or release documentation with the completed slice
- [x] Task: Run validation gates
    - [x] Run focused diagnostics and schema tests
    - [x] Run relevant lint, type, and documentation checks
    - [x] Confirm bindings fixtures still load diagnostics artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Gates' (Protocol in workflow.md)
