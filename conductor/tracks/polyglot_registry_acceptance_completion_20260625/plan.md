# Polyglot Registry Acceptance Completion Plan

## Phase 1: Registry State Audit

- [ ] Task: Inventory language and HPC registry states
    - [ ] Audit Python, TestPyPI/PyPI, conda, Rust, R, Julia, TypeScript, Go, C#, Spack, EasyBuild, HPSF, and E4S artifacts.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write receipt-state consistency tests
    - [ ] Add tests that require accepted/submitted/deferred states to match receipts and owner-backed deferrals.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Registry State Audit' (Protocol in workflow.md)

## Phase 2: Local Package Evidence Refresh

- [ ] Task: Refresh language package dry-runs
    - [ ] Run or record Python, Rust, R, Julia, TypeScript, Go, and C# package dry-runs and smoke gates.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Refresh HPC packaging evidence
    - [ ] Validate Spack/EasyBuild templates, Python 3.14 constraints, install smoke logs, and submission packet evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Local Package Evidence Refresh' (Protocol in workflow.md)

## Phase 3: External Acceptance Ledger

- [ ] Task: Reconcile receipts and deferrals
    - [ ] Update registry receipts, target inventory, and external acceptance deferrals.
    - [ ] Ensure no target is marked accepted without a receipt URL or artifact.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update release and Starlight pages
    - [ ] Align public language with actual registry state.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: External Acceptance Ledger' (Protocol in workflow.md)

## Phase 4: Review, Push, and CI

- [ ] Task: Run registry validation suite
    - [ ] Run targeted registry, package, docs, and release-readiness tests.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run final conductor-review and GitHub Actions monitoring
    - [ ] Apply fixes, push, and monitor CI until green or externally blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
