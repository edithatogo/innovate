# Implementation Plan: Vision and Roadmap Truth Audit

## Phase 1: Red-Phase Status Coverage [checkpoint: 4645c52]

- [x] Task: Inventory current vision and roadmap completion claims b1cb1a5
    - [x] Review `conductor/product.md`, `conductor/tech-stack.md`, `docs/architecture_modernization_roadmap.md`, `docs/source/rust_core_roadmap.rst`, and Astro/Starlight operations pages
    - [x] Classify each claim as implemented, archived-track-complete, future-state, blocked, or stale
    - [x] Commit this task before starting the next task
- [x] Task: Write failing tests for stale completion claims 387848e
    - [x] Add tests that reject "full Rust core complete" claims while the inventory has Python-owned or bridge-owned slices
    - [x] Add tests that reject product-status documentation claiming Sphinx as the active docs stack
    - [x] Add tests that require roadmap pages to point unresolved work to active Conductor tracks
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Red-Phase Status Coverage' (Protocol in workflow.md)

## Phase 2: Canonical Status Documentation

- [x] Task: Update product and roadmap status language 53198f2
    - [x] Add a canonical status statement for completed tracks versus incomplete future-state vision
    - [x] Link each future-state boundary to a granular track
    - [x] Remove or qualify stale Sphinx and full-completion wording
    - [x] Commit this task before starting the next task
- [~] Task: Update Astro/Starlight docs mirrors
    - [~] Ensure the docs site surfaces the same canonical status wording
    - [ ] Preserve links to archival evidence and future-state tracks
    - [ ] Commit this task before starting the next task
- [ ] Task: Run targeted tests and docs checks
    - [ ] Run the new roadmap-status tests
    - [ ] Run any existing roadmap/doc architecture tests that cover these files
    - [ ] Commit validation-only changes if needed
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Canonical Status Documentation' (Protocol in workflow.md)

## Phase 3: Final Track Review

- [ ] Task: Reconcile all status matrices
    - [ ] Confirm no roadmap page has an unowned gap
    - [ ] Confirm no page overstates implementation, submission, or cutover status
    - [ ] Commit this task before starting the next task
- [ ] Task: Run final conductor review
    - [ ] Review the full track diff against the spec, plan, workflow, and tests
    - [ ] Apply any high-confidence fixes
    - [ ] Re-run validation until stable
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Track Review' (Protocol in workflow.md)
