# Implementation Plan

## Phase 1: Conformance Contract and Golden Fixtures

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
- [~] Task: Conductor - User Manual Verification 'Conformance Contract and Golden Fixtures' (Protocol in workflow.md)

## Phase 2: Language Binding Hardening

- [ ] Task: Harden R, Julia, and TypeScript bindings
    - [ ] Add or refresh language-native package checks
    - [ ] Add idiomatic examples and docs snippets
    - [ ] Add conformance evidence for supported operations
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Harden Go, C#, and Rust bindings
    - [ ] Add or refresh package checks, examples, and version validation
    - [ ] Add conformance evidence for supported operations
    - [ ] Align errors and serialization with the shared contract
    - [ ] Commit implementation changes for this task
    - [ ] Commit this plan update with the task commit SHA
- [ ] Task: Conductor - User Manual Verification 'Language Binding Hardening' (Protocol in workflow.md)

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
