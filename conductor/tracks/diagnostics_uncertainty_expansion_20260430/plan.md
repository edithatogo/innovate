# Implementation Plan: Diagnostics and Uncertainty Expansion

## Phase 1: Artifact Contract

- [ ] Task: Inventory diagnostics and uncertainty outputs
    - [ ] List existing diagnostic functions, summaries, and model-comparison helpers
    - [ ] Map outputs to schema and Arrow interchange support
    - [ ] Identify diagnostics currently tied to private Python objects
- [ ] Task: Define the richer diagnostics contract
    - [ ] Specify artifact payloads for residuals, calibration, uncertainty, and model comparison
    - [ ] Evaluate JAX/XLA eligibility for array-heavy diagnostics and uncertainty summaries
    - [ ] Define support tiers and promotion criteria by model family
    - [ ] Document compatibility expectations for bindings
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Artifact Contract' (Protocol in workflow.md)

## Phase 2: Implementation Slice

- [ ] Task: Add diagnostics contract tests
    - [ ] Write failing tests for selected diagnostic artifact schemas
    - [ ] Add representative cross-language fixture payloads
    - [ ] Add tolerance-based checks for stochastic or approximate outputs
- [ ] Task: Implement selected richer diagnostics
    - [ ] Add the minimal diagnostics logic for the chosen model family or workflow
    - [ ] Use a JAX/XLA-backed path for eligible kernels or document why the selected diagnostics are not XLA-suitable
    - [ ] Emit stable schema-compatible artifact payloads
    - [ ] Preserve current public API compatibility
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Implementation Slice' (Protocol in workflow.md)

## Phase 3: Documentation and Gates

- [ ] Task: Update diagnostics documentation
    - [ ] Document supported artifacts, model families, and interpretation notes
    - [ ] Add examples for consuming diagnostics through kernel payloads
    - [ ] Update roadmap or release documentation with the completed slice
- [ ] Task: Run validation gates
    - [ ] Run focused diagnostics and schema tests
    - [ ] Run relevant lint, type, and documentation checks
    - [ ] Confirm bindings fixtures still load diagnostics artifacts
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Gates' (Protocol in workflow.md)
