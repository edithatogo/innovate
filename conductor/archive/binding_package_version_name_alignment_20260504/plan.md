# Implementation Plan: Binding Package Version and Language-Suffix Name Alignment

## Phase 1: Metadata Alignment

- [x] Task: Add package naming and version alignment guards
    - [x] Check binding package names and versions against the shared release.
    - [x] Preserve documented registry-valid exceptions for Rust and Julia.
- [x] Task: Update binding package metadata
    - [x] Align TypeScript, Rust, R, Julia, and C# versions to `0.5.0`.
    - [x] Update package names and publication metadata.
- [x] Task: Update publication docs and workflow checks
    - [x] Document package manager targets and naming constraints.
    - [x] Update NuGet metadata validation.
- [x] Task: Run package dry-run validation
    - [x] Validate TypeScript tests and `npm pack --dry-run`.
    - [x] Validate Rust tests and `cargo package`.
    - [x] Validate R build/check.
    - [x] Validate Julia `Pkg.test()`.
    - [x] Validate Go tests.
    - [x] Validate local C# net10.0 pack metadata.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Metadata Alignment' (Protocol in workflow.md)
