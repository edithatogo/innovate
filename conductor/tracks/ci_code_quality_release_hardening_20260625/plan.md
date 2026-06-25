# CI, Code Quality, and Release Hardening Plan

## Phase 1: Gate Inventory and Failing Tests

- [x] Task: Inventory current local and CI gates
    - [x] Map nox sessions, GitHub Actions, release evidence, dependency dashboards, security, mutation, and coverage outputs.
    - [x] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write release-hardening guard tests
    - [ ] Add tests for evidence freshness, required gate presence, and release-ready fail-closed behavior.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Gate Inventory and Failing Tests' (Protocol in workflow.md)

## Phase 2: CI and Automation Hardening

- [ ] Task: Align nox and GitHub Actions gates
    - [ ] Ensure lint, types, tests, docs, package, security, dependency dashboards, and binding checks are covered.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add mutation and coverage release evidence
    - [ ] Update mutation testing and coverage workflows/evidence with clear thresholds and schedules.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: CI and Automation Hardening' (Protocol in workflow.md)

## Phase 3: Observability and Release Evidence

- [ ] Task: Harden runtime and release observability
    - [ ] Validate structured logging, release reports, SBOM, provenance, checksums, and security evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Refresh release-readiness report
    - [ ] Require fresh passing evidence before `release_ready`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Observability and Release Evidence' (Protocol in workflow.md)

## Phase 4: Full Release Gate

- [ ] Task: Run full local release validation
    - [ ] Run `uv run nox -s lint types tests docs package`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review, push, and monitor CI
    - [ ] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Full Release Gate' (Protocol in workflow.md)
