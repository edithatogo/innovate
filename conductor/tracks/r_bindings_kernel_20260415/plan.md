# Implementation Plan: R Bindings over the Functional Kernel

## Phase 1: Binding Architecture and Red-Phase Tests [checkpoint: 5892b39]

- [x] Task: Define the R binding architecture [e13f763]
    - [ ] Choose the R package layout and invocation path into the kernel
    - [ ] Define mapping rules between kernel schemas and R objects
    - [ ] Write failing tests for the basic R-to-kernel contract
- [x] Task: Scaffold the R package structure [e13f763]
    - [ ] Add package metadata and development scaffolding
    - [ ] Add test harness support for contract validation
    - [ ] Confirm the new R binding tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Binding Architecture and Red-Phase Tests' (Protocol in workflow.md)

## Phase 2: Stable Operation Wrappers [checkpoint: 96ba636]

- [x] Task: Implement R wrappers for stable kernel operations [fcb7e91]
    - [x] Add discovery, fit, predict, and summarize wrappers
    - [x] Add error mapping and structured result conversion
    - [x] Make the wrapper tests pass
- [x] Task: Add diagnostics and example workflows [c2f857c]
    - [x] Expose stable diagnostics surfaces through idiomatic R helpers
    - [x] Add at least one end-to-end example
    - [x] Validate parity with the kernel contract
- [x] Task: Conductor - User Manual Verification 'Phase 2: Stable Operation Wrappers' (Protocol in workflow.md) [96ba636]

## Phase 3: Packaging and User Guidance

- [x] Task: Harden the R package for users and CI [d15dbfb]
    - [x] Add packaging guidance and installation notes
    - [x] Add automated checks for schema-compatibility drift
    - [x] Verify core examples run in automated test contexts
- [ ] Task: Document the R binding surface
    - [ ] Add user-facing docs and API guidance
    - [ ] Document backend expectations and support boundaries
    - [ ] Verify all acceptance criteria are satisfied
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Packaging and User Guidance' (Protocol in workflow.md)
