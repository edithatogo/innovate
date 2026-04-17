# Implementation Plan: Network and Policy Diffusion

## Phase 1: Data Contracts and Red-Phase Tests

- [~] Task: Define network and policy input contracts
    - [ ] Specify accepted adjacency, edge-list, and event-timing data shapes
    - [ ] Define canonical validation rules and error handling
    - [ ] Write failing tests for the input contracts
- [ ] Task: Add shared fixtures for spillover and timing scenarios
    - [ ] Create representative synthetic network datasets
    - [ ] Create representative policy-adoption timing datasets
    - [ ] Confirm the new tests fail before implementation
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Data Contracts and Red-Phase Tests' (Protocol in workflow.md)

## Phase 2: Implement Core Models

- [ ] Task: Implement a network-aware diffusion model
    - [ ] Add fitting and prediction logic for contagion or peer-effect structure
    - [ ] Expose interpretable spillover summaries
    - [ ] Make the network-model tests pass
- [ ] Task: Implement a policy or hazard-based diffusion model
    - [ ] Add event-history or timing-sensitive model logic
    - [ ] Expose timing-effect summaries and forecasts
    - [ ] Make the policy-model tests pass
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implement Core Models' (Protocol in workflow.md)

## Phase 3: Validation and User Guidance

- [ ] Task: Integrate diagnostics and capability metadata
    - [ ] Register the new model families in the capability registry
    - [ ] Add diagnostics for spillover, timing, and fit quality
    - [ ] Verify compatibility with canonical APIs
- [ ] Task: Document network and policy workflows
    - [ ] Add usage examples and data-shape guidance
    - [ ] Document assumptions and interpretation caveats
    - [ ] Verify all acceptance criteria are satisfied
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Validation and User Guidance' (Protocol in workflow.md)
