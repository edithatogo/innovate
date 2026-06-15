# Implementation Plan

## Phase 1: Conformance Contract and Golden Fixtures [checkpoint: 4212538]

- [x] Task: Define cross-language conformance schema [7c36274]
    - [x] Specify binding status, capability, operation, payload, error, and version fields
    - [x] Add tests that fail on missing binding evidence
    - [x] Generate a machine-readable binding conformance inventory
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Add golden fixture suite [32a3040]
    - [x] Create canonical operation and payload fixtures
    - [x] Add expected outputs with documented numerical tolerances
    - [x] Wire fixture validation into Python and Rust first
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Conformance Contract and Golden Fixtures' (Protocol in workflow.md)

## Phase 2: Language Binding Hardening [checkpoint: d84beb1]

- [x] Task: Harden R, Julia, and TypeScript bindings [b02d318]
    - [x] Add or refresh language-native package checks
    - [x] Add idiomatic examples and docs snippets
    - [x] Add conformance evidence for supported operations
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Harden Go, C#, and Rust bindings [308e21b]
    - [x] Add or refresh package checks, examples, and version validation
    - [x] Add conformance evidence for supported operations
    - [x] Align errors and serialization with the shared contract
    - [x] Commit implementation changes for this task
    - [x] Commit this plan update with the task commit SHA
- [x] Task: Conductor - User Manual Verification 'Language Binding Hardening' (Protocol in workflow.md)

## Phase 3: CI, Docs, and Release Evidence

- [ ] Task: Add binding conformance CI gates
    - [ ] Add or update GitHub Actions workflows for language-native checks
    - [ ] Upload conformance reports as CI artifacts
    - [ ] Ensure local checks have documented fallbacks when toolchains are unavailable
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Publish binding parity documentation
    - [ ] Add Starlight binding parity pages
    - [ ] Link package-manager receipts and conformance evidence
    - [ ] Add stale-claim tests for docs and package manifests
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'CI, Docs, and Release Evidence' (Protocol in workflow.md)
