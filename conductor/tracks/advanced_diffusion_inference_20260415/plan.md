# Implementation Plan: Advanced Diffusion Inference

## Phase 1: Shared Advanced-Model Foundations

- [x] Task: Define the advanced-model contract [d91f2db]
    - [x] Specify canonical interfaces for fit, predict, simulate, and summarize
    - [x] Add capability metadata for advanced probabilistic models
    - [x] Write failing tests for the shared contract
- [x] Task: Prepare backend-aware fixtures and test scaffolding [d91f2db]
    - [x] Add fixtures for optional probabilistic dependencies
    - [x] Add representative synthetic datasets for advanced inference cases
    - [x] Confirm the new tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Shared Advanced-Model Foundations' (Protocol in workflow.md) [d91f2db]

## Phase 2: Hierarchical and Latent-Process Models

- [ ] Task: Implement hierarchical diffusion support
    - [ ] Add grouped or partially pooled parameter handling
    - [ ] Expose posterior or uncertainty summaries
    - [ ] Make the hierarchical tests pass
- [ ] Task: Implement a latent-process diffusion workflow
    - [ ] Add a state-space or equivalent latent-process model
    - [ ] Expose forecasting outputs with uncertainty
    - [ ] Make the latent-process tests pass
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Hierarchical and Latent-Process Models' (Protocol in workflow.md)

## Phase 3: Structural Breaks and Documentation

- [ ] Task: Implement change-point or regime-switching diffusion support
    - [ ] Add a structural-break model variant
    - [ ] Add tests for break detection or regime transitions
    - [ ] Validate comparisons against simpler baselines
- [ ] Task: Document advanced model usage
    - [ ] Add examples for hierarchical and change-point analysis
    - [ ] Document backend and installation requirements
    - [ ] Verify all acceptance criteria are satisfied
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Structural Breaks and Documentation' (Protocol in workflow.md)
