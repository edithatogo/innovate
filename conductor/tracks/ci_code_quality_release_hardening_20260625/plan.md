# CI, Code Quality, and Release Hardening Plan

## Phase 1: Gate Inventory and Failing Tests

- [x] Task: Inventory current local and CI gates
    - [x] Map nox sessions, GitHub Actions, release evidence, dependency dashboards, security, mutation, and coverage outputs.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write release-hardening guard tests
    - [x] Add tests for evidence freshness, required gate presence, and release-ready fail-closed behavior.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Gate Inventory and Failing Tests' (Protocol in workflow.md)
    - [x] All 20 guard tests pass, gates inventoried, phase complete.

## Phase 2: CI and Automation Hardening

- [x] Task: Align nox and GitHub Actions gates
    - [x] Ensure lint, types, tests, docs, package, security, dependency dashboards, and binding checks are covered.
    - [x] Add `coverage`, `mutation`, `dependency_dashboard`, and `binding_conformance` nox sessions.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add mutation and coverage release evidence
    - [x] Update mutation testing and coverage workflows/evidence with clear thresholds and schedules.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: CI and Automation Hardening' (Protocol in workflow.md)
    - [x] Verified by conductor-review: plan.md updated, gate-inventory.md consistent, 5 new tests pass, git note attached, nox sessions aligned.

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
