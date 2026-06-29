# Roadmap Release Truth Closure Plan

## Phase 1: Roadmap and Evidence Inventory

- [~] Task: Build roadmap source inventory
    - [x] Enumerate roadmap docs, ADRs, product vision docs, Conductor registry entries, archived tracks, and static evidence artifacts.
    - [x] Commit inventory changes and attach the required Conductor git note.
- [x] Task: Write failing truth-ledger coverage tests
    - [x] Add tests that require each roadmap claim to map to evidence, active track, external blocker, or out-of-scope rationale.
    - [x] Confirm tests fail before the ledger exists.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Roadmap and Evidence Inventory' (Protocol in workflow.md)

## Phase 2: Truth Ledger Implementation

- [x] Task: Create machine-readable roadmap truth ledger
    - [x] Record status, owner, evidence, track link, blocker state, and release-claim policy for every item.
    - [x] Include Rust core, Starlight docs, dependency, CI, release, external registry, and advanced modeling gaps.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Reconcile public docs and release evidence
    - [x] Update roadmap, release readiness, and Starlight pages to reference the ledger.
    - [x] Remove or qualify any claim that lacks evidence.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Truth Ledger Implementation' (Protocol in workflow.md)

## Phase 3: Final Verification and CI

- [x] Task: Run full local validation
    - [x] Run `uv run nox -s lint types tests docs package`.
    - [x] Fix failures or record explicit blockers.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run conductor-review and apply findings
    - [x] Run the conductor-review skill for this track.
    - [x] Apply findings, rerun validation, and commit fixes.
    - [x] Push and monitor GitHub Actions until all triggered checks pass.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Final Verification and CI' (Protocol in workflow.md)
