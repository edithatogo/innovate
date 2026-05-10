# Implementation Plan: Network and Policy Diffusion

## Phase 1: Data Contracts and Red-Phase Tests [checkpoint: c330b84]

- [x] Task: Define network and policy input contracts [3d622f3]
    - [x] Specify accepted adjacency, edge-list, and event-timing data shapes
    - [x] Define canonical validation rules and error handling
    - [x] Write failing tests for the input contracts
- [x] Task: Add shared fixtures for spillover and timing scenarios [226f638]
    - [x] Create representative synthetic network datasets
    - [x] Create representative policy-adoption timing datasets
    - [x] Confirm the new tests fail before implementation
- [x] Task: Conductor - User Manual Verification 'Phase 1: Data Contracts and Red-Phase Tests' (Protocol in workflow.md) [c330b84]

## Phase 2: Implement Core Models [checkpoint: 3d622f3]

- [x] Task: Implement a network-aware diffusion model [3d622f3]
    - [x] Add fitting and prediction logic for contagion or peer-effect structure
    - [x] Expose interpretable spillover summaries
    - [x] Make the network-model tests pass
- [x] Task: Implement a policy or hazard-based diffusion model [3d622f3]
    - [x] Add event-history or timing-sensitive model logic
    - [x] Expose timing-effect summaries and forecasts
    - [x] Make the policy-model tests pass
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implement Core Models' (Protocol in workflow.md) [3d622f3]

## Phase 3: Validation and User Guidance [checkpoint: 3d622f3]

- [x] Task: Integrate diagnostics and capability metadata [3d622f3]
    - [x] Register the new model families in the capability registry
    - [x] Add diagnostics for spillover, timing, and fit quality
    - [x] Verify compatibility with canonical APIs
- [x] Task: Document network and policy workflows [3d622f3]
    - [x] Add usage examples and data-shape guidance
    - [x] Document assumptions and interpretation caveats
    - [x] Verify all acceptance criteria are satisfied
- [x] Task: Conductor - User Manual Verification 'Phase 3: Validation and User Guidance' (Protocol in workflow.md) [3d622f3]
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and User Guidance' (Protocol in workflow.md) [3d622f3]
