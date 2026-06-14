# Implementation Plan: External Submission Blocker Closure

## Phase 1: Submission State Red Tests [checkpoint: 4402623]

- [x] Task: Inventory external submission targets bb962de
    - [x] Parse registry receipts, HPC packets, community matrices, and governance notes
    - [x] Identify generic blocked states and maintainer-managed actions
    - [x] Commit this task before starting the next task
- [x] Task: Write failing submission-state tests 1bc7818
    - [x] Require each target to have status, owner, evidence, and next action
    - [x] Reject submitted or accepted claims without receipts
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Submission State Red Tests' (Protocol in workflow.md)

## Phase 2: Evidence Refresh

- [x] Task: Refresh package-manager target evidence
    - [x] Update statuses for npm, crates.io, R-universe/CRAN, Julia General, Go modules, and NuGet
    - [x] Record credential-blocked or maintainer-ready states explicitly
    - [x] Commit this task before starting the next task
- [ ] Task: Refresh HPC target evidence
    - [ ] Update Spack, EasyBuild, HPSF, and E4S packets and blocker notes
    - [ ] Preserve scheduler and environment probe evidence
    - [ ] Commit this task before starting the next task
- [ ] Task: Refresh scientific community dossiers
    - [ ] Update pyOpenSci, rOpenSci, JOSS, NumFOCUS, PyPA, Apache Arrow, .NET Foundation, Julia, and R community states
    - [ ] Record submission readiness versus actual submission state
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Evidence Refresh' (Protocol in workflow.md)

## Phase 3: Docs and Packet Regeneration

- [ ] Task: Regenerate machine-readable packets
    - [ ] Rebuild registry receipt, HPC packet, community matrix, and Astro/Starlight mirror artifacts
    - [ ] Commit this task before starting the next task
- [ ] Task: Update docs and roadmap wording
    - [ ] Remove overclaims and stale blocker language
    - [ ] Link every target to current evidence
    - [ ] Commit this task before starting the next task
- [ ] Task: Run final conductor review
    - [ ] Review the full track diff against the spec, plan, workflow, and tests
    - [ ] Apply high-confidence fixes and rerun validation
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Docs and Packet Regeneration' (Protocol in workflow.md)
