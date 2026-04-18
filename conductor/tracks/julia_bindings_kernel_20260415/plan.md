# Implementation Plan: Julia Bindings over the Functional Kernel

## Phase 1: Binding Architecture and Red-Phase Tests [checkpoint: 74eff33]

- [x] Task: Define the Julia binding architecture [6d84a63]
    - [x] Choose the Julia package layout and invocation path into the kernel
    - [x] Define mapping rules between kernel schemas and Julia types
    - [x] Write failing tests for the basic Julia-to-kernel contract
- [x] Task: Scaffold the Julia package structure [6d84a63]
    - [x] Add package metadata and development scaffolding
    - [x] Add test harness support for contract validation
    - [x] Confirm the new Julia binding tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Binding Architecture and Red-Phase Tests' (Protocol in workflow.md) [74eff33]

## Phase 2: Stable Operation Wrappers [checkpoint: 8d4d44e]

- [x] Task: Implement Julia wrappers for stable kernel operations
    - [x] Add discovery, fit, predict, and summarize wrappers
    - [x] Add error mapping and structured result conversion
    - [x] Make the wrapper tests pass
- [x] Task: Add diagnostics and example workflows
    - [x] Expose stable diagnostics surfaces through idiomatic Julia helpers
    - [x] Add at least one end-to-end example
    - [x] Validate parity with the kernel contract
- [x] Task: Conductor - User Manual Verification 'Phase 2: Stable Operation Wrappers' (Protocol in workflow.md) [fcf34fa]

## Phase 3: Packaging and User Guidance

- [x] Task: Harden the Julia package for users and CI [4b38790]
    - [x] Add packaging guidance and installation notes
    - [x] Add automated checks for schema-compatibility drift
    - [x] Verify core examples run in automated test contexts
- [x] Task: Document the Julia binding surface [4b38790]
    - [x] Add user-facing docs and API guidance
    - [x] Document backend expectations and support boundaries
    - [x] Verify all acceptance criteria are satisfied
- [~] Task: Conductor - User Manual Verification 'Phase 3: Packaging and User Guidance' (Protocol in workflow.md)
