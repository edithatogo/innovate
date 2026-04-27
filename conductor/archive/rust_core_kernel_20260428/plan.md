# Implementation Plan: Rust Core Kernel Roadmap and C# Binding Foundation

## Phase 1: Rust Core Migration Scope

- [x] Task: Define the Rust core migration boundary
    - [x] Inventory current kernel operations and schemas
    - [x] Identify operations suitable for initial Rust-backed execution
    - [x] Document operations that must remain Python-backed initially
- [x] Task: Add Rust parity and benchmark expectations
    - [x] Define Python reference semantics for candidate operations
    - [x] Specify parity tests for Rust-backed behavior
    - [x] Specify benchmark gates for promoting Rust-backed execution
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Core Migration Scope' (Protocol in workflow.md)

## Phase 2: C# Binding Foundation

- [x] Task: Define the C# binding architecture
    - [x] Choose the C# package layout and invocation path into the kernel
    - [x] Define mapping rules between kernel schemas and C# objects
    - [x] Write failing tests for the basic C#-to-kernel contract
- [x] Task: Scaffold C# binding documentation and validation hooks
    - [x] Add C# binding user guidance and support boundaries
    - [x] Add schema-compatibility validation expectations
    - [x] Confirm C# binding scope does not duplicate model logic
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: C# Binding Foundation' (Protocol in workflow.md)

## Phase 3: Documentation and Governance

- [x] Task: Synchronize architecture documentation
    - [x] Update roadmap documentation with Rust-core sequencing
    - [x] Cross-link ADR 0004 from relevant architecture docs
    - [x] Document the thin-binding policy for all language surfaces
- [x] Task: Add drift-prevention checks
    - [x] Identify validation checks that guard schema compatibility
    - [x] Add or update tests that fail when bindings diverge from the kernel contract
    - [x] Verify acceptance criteria are satisfied
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Governance' (Protocol in workflow.md)
