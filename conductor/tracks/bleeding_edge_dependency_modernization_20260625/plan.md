# Bleeding-Edge Dependency Modernization Plan

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Dependency Baseline Audit

- [x] Task: Inventory all ecosystem dependencies
    - [x] Run `uv tree --outdated`, `pnpm outdated`, `npm outdated`, Cargo checks, and R/Julia/.NET package checks where available.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write dependency baseline tests
    - [x] Add tests for Python 3.14, NumPy 2+, Pydantic v2, basedpyright strict, Astro 7, TypeScript 6, Node 26 types, Vitest 4, criterion 0.8, and mutmut current baseline.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependency Baseline Audit' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Runtime and Manifest Modernization

- [x] Task: Align Python and scientific dependency metadata
    - [x] Update pyproject, uv lock, nox, CI, and docs for the selected floors and upper bounds.
    - [x] Replace or constrain pandas usage toward Polars where feasible.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Align frontend, Rust, and binding toolchains
    - [x] Update Astro/Starlight, TypeScript, Vitest, Node types, Rust benchmark dependencies, and binding manifests.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Runtime and Manifest Modernization' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Dashboard Automation

- [x] Task: Implement dependency dashboard workflow
    - [x] Ensure all ecosystem outdated checks produce artifacts without requiring credentials.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Wire release-readiness evidence
    - [x] Make dependency dashboard status part of release readiness.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Dashboard Automation' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: pending]

## Phase 4: Review, Push, and CI

- [x] Task: Run full modernization validation
    - [x] Run `uv run nox -s lint types tests docs package` plus ecosystem package checks.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run final conductor-review and CI monitor
    - [x] Apply fixes, push, and monitor GitHub Actions until green or blocked.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
