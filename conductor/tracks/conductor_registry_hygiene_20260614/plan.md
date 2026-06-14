# Implementation Plan: Conductor Registry Hygiene

## Phase 1: Registry Drift Red Tests [checkpoint: a583530]

- [x] Task: Inventory registry and filesystem drift aceab82
    - [x] Compare `conductor/tracks.md`, `conductor/tracks/`, and `conductor/archive/`
    - [x] Identify stale active folders, missing archive folders, and broken links
    - [x] Commit this task before starting the next task
- [x] Task: Write failing registry hygiene tests 21d200d
    - [x] Require active folders to have active registry entries
    - [x] Require completed registry entries to point into `conductor/archive/`
    - [x] Require status tooling to report stale or orphaned folders
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Registry Drift Red Tests' (Protocol in workflow.md)

## Phase 2: Registry Reconciliation [checkpoint: a569237]

- [x] Task: Reconcile stale active track folders 2bd7c19
    - [x] Move completed active folders to archive or remove duplicates only after preserving evidence
    - [x] Update links and metadata as needed
    - [x] Commit this task before starting the next task
- [x] Task: Add registry hygiene automation 96f0216
    - [x] Add or update a script/test that checks registry-to-filesystem consistency
    - [x] Include stale-folder diagnostics in the status output or docs
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Registry Reconciliation' (Protocol in workflow.md)

## Phase 3: Final Hygiene Gate

- [~] Task: Run full registry hygiene validation
    - [~] Verify active tracks, archived tracks, and registry links
    - [~] Verify no completed work remains in active track directories
    - [ ] Commit this task before starting the next task
- [ ] Task: Run final conductor review
    - [ ] Review the full track diff against the spec, plan, workflow, and tests
    - [ ] Apply high-confidence fixes and rerun validation
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Hygiene Gate' (Protocol in workflow.md)
