# Implementation Plan: Binding Publication and Multi-Language CI

## Phase 1: Publication Plan

- [x] Task: Document package-manager targets
    - [x] Add npm target for TypeScript
    - [x] Add crates.io target for Rust
    - [x] Add R-universe/CRAN target for R
    - [x] Add Julia General target for Julia
    - [x] Add versioned Go module target for Go
    - [x] Add planned NuGet target for C#
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Publication Plan' (Protocol in workflow.md)

## Phase 2: Multi-Language CI

- [x] Task: Add binding CI jobs
    - [x] Add Rust cargo fmt and test job
    - [x] Add TypeScript schema, typecheck, and test job
    - [x] Add Go test job
    - [x] Add Julia instantiate and test job
    - [x] Add R dependency install and test job
- [x] Task: Add release-gated publication workflow
    - [x] Add npm package checks and publish hook
    - [x] Add crates.io package checks and publish hook
    - [x] Add R package build/check gate
    - [x] Add Julia package test gate
    - [x] Add Go module release gate
    - [x] Add planned NuGet gate for C#
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Multi-Language CI' (Protocol in workflow.md)

## Phase 3: Validation

- [x] Task: Add regression tests
    - [x] Test binding publication docs list registry targets
    - [x] Test main CI includes implemented binding jobs
    - [x] Test publication workflow includes release-gated registry steps
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation' (Protocol in workflow.md)
