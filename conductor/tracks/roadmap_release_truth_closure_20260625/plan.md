# Roadmap Release Truth Closure Plan

## Phase 1: Roadmap and Evidence Inventory

- [ ] Task: Build roadmap source inventory
    - [ ] Enumerate roadmap docs, ADRs, product vision docs, Conductor registry entries, archived tracks, and static evidence artifacts.
    - [ ] Commit inventory changes and attach the required Conductor git note.
- [ ] Task: Write failing truth-ledger coverage tests
    - [ ] Add tests that require each roadmap claim to map to evidence, active track, external blocker, or out-of-scope rationale.
    - [ ] Confirm tests fail before the ledger exists.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Roadmap and Evidence Inventory' (Protocol in workflow.md)

## Phase 2: Truth Ledger Implementation

- [ ] Task: Create machine-readable roadmap truth ledger
    - [ ] Record status, owner, evidence, track link, blocker state, and release-claim policy for every item.
    - [ ] Include Rust core, Starlight docs, dependency, CI, release, external registry, and advanced modeling gaps.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Reconcile public docs and release evidence
    - [ ] Update roadmap, release readiness, and Starlight pages to reference the ledger.
    - [ ] Remove or qualify any claim that lacks evidence.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Truth Ledger Implementation' (Protocol in workflow.md)

## Phase 3: Final Verification and CI

- [ ] Task: Run full local validation
    - [ ] Run `uv run nox -s lint types tests docs package`.
    - [ ] Fix failures or record explicit blockers.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review and apply findings
    - [ ] Run the conductor-review skill for this track.
    - [ ] Apply findings, rerun validation, and commit fixes.
    - [ ] Push and monitor GitHub Actions until all triggered checks pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Verification and CI' (Protocol in workflow.md)
