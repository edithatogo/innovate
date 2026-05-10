# Implementation Plan: Arrow Interchange and Schema Layer

## Phase 1: Schema Definition and Red-Phase Tests

- [x] Task: Define the interchange surface
    - [x] Inventory the kernel payloads that need Arrow-compatible schemas
    - [x] Specify versioned schema shapes for tabular outputs, diagnostics, and metadata
    - [x] Define the pandas-to-Arrow mapping rules for the stable contract
- [x] Task: Write failing interchange tests
    - [x] Add schema validation tests for the documented payloads
    - [x] Add round-trip tests for pandas and PyArrow conversions
    - [x] Confirm the interchange tests fail in the red phase
- [x] Task: Conductor - User Manual Verification 'Phase 1: Schema Definition and Red-Phase Tests' (Protocol in workflow.md)

## Phase 2: Python Adapters and Kernel Integration

- [x] Task: Implement the Python interchange adapters
    - [x] Add canonical conversion helpers for the stable payloads
    - [x] Add validation hooks for schema version and payload correctness
    - [x] Make the interchange tests pass
- [x] Task: Integrate the interchange layer with the kernel boundary
    - [x] Ensure kernel payload documentation references the Arrow layer
    - [x] Expose machine-readable schema metadata where appropriate
    - [x] Verify that Python-facing ergonomics remain acceptable
- [x] Task: Conductor - User Manual Verification 'Phase 2: Python Adapters and Kernel Integration' (Protocol in workflow.md)

## Phase 3: Documentation and Binding Guidance

- [x] Task: Document the interchange layer
    - [x] Add user-facing architecture documentation for the Arrow boundary
    - [x] Add binding-author guidance for versioning and compatibility
    - [x] Document the role of pandas, PyArrow, and selective Polars usage
- [x] Task: Verify readiness for downstream work
    - [x] Map the interchange layer to the functional kernel and binding tracks
    - [x] Confirm the acceptance criteria are satisfied
    - [x] Record any deferred schema areas explicitly
- [x] Task: Conductor - User Manual Verification 'Phase 3: Documentation and Binding Guidance' (Protocol in workflow.md)
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Binding Guidance' (Protocol in workflow.md)
