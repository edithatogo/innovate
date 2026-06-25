# Bleeding-Edge Dependency Modernization Plan

## Phase 1: Dependency Baseline Audit

- [ ] Task: Inventory all ecosystem dependencies
    - [ ] Run `uv tree --outdated`, `pnpm outdated`, `npm outdated`, Cargo checks, and R/Julia/.NET package checks where available.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write dependency baseline tests
    - [ ] Add tests for Python 3.14, NumPy 2+, Pydantic v2, basedpyright strict, Astro 7, TypeScript 6, Node 26 types, Vitest 4, criterion 0.8, and mutmut current baseline.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Dependency Baseline Audit' (Protocol in workflow.md)

## Phase 2: Runtime and Manifest Modernization

- [ ] Task: Align Python and scientific dependency metadata
    - [ ] Update pyproject, uv lock, nox, CI, and docs for the selected floors and upper bounds.
    - [ ] Replace or constrain pandas usage toward Polars where feasible.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Align frontend, Rust, and binding toolchains
    - [ ] Update Astro/Starlight, TypeScript, Vitest, Node types, Rust benchmark dependencies, and binding manifests.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Runtime and Manifest Modernization' (Protocol in workflow.md)

## Phase 3: Dashboard Automation

- [ ] Task: Implement dependency dashboard workflow
    - [ ] Ensure all ecosystem outdated checks produce artifacts without requiring credentials.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Wire release-readiness evidence
    - [ ] Make dependency dashboard status part of release readiness.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Dashboard Automation' (Protocol in workflow.md)

## Phase 4: Review, Push, and CI

- [ ] Task: Run full modernization validation
    - [ ] Run `uv run nox -s lint types tests docs package` plus ecosystem package checks.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run final conductor-review and CI monitor
    - [ ] Apply fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
