# Implementation Plan: Advanced Diffusion Inference

## Phase 1: Shared Advanced-Model Foundations [checkpoint: 1aeffb5]

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

- [x] Task: Implement hierarchical diffusion support [d91f2db]
    - [x] Add grouped or partially pooled parameter handling
    - [x] Expose posterior or uncertainty summaries
    - [x] Make the hierarchical tests pass
- [x] Task: Implement a latent-process diffusion workflow [d91f2db]
    - [x] Add a state-space or equivalent latent-process model
    - [x] Expose forecasting outputs with uncertainty
    - [x] Make the latent-process tests pass
- [x] Task: Conductor - User Manual Verification 'Phase 2: Hierarchical and Latent-Process Models' (Protocol in workflow.md) [d91f2db]

## Phase 3: Structural Breaks and Documentation

- [x] Task: Implement change-point or regime-switching diffusion support [d91f2db]
    - [x] Add a structural-break model variant
    - [x] Add tests for break detection or regime transitions
    - [x] Validate comparisons against simpler baselines
- [x] Task: Document advanced model usage [609f3ea]
    - [x] Add examples for hierarchical and change-point analysis
    - [x] Document backend and installation requirements
    - [x] Verify all acceptance criteria are satisfied
- [x] Task: Conductor - User Manual Verification 'Phase 3: Structural Breaks and Documentation' (Protocol in workflow.md) [609f3ea]
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Structural Breaks and Documentation' (Protocol in workflow.md)
