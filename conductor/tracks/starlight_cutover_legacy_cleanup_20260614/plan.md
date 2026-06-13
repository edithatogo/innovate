# Implementation Plan: Starlight Cutover and Legacy Cleanup

## Phase 1: Red-Phase Cutover Checks [checkpoint: 8135df1]

- [x] Task: Inventory active and legacy docs surfaces babb590
    - [x] Compare product, tech-stack, Starlight config, Sphinx config, and migration manifests
    - [x] Identify stale migration-in-progress language
    - [x] Commit this task before starting the next task
- [x] Task: Write failing cutover tests a2e9d4b
    - [x] Reject product docs claiming Sphinx is the active docs stack
    - [x] Reject active track folders for completed Starlight migration work
    - [x] Require legacy Sphinx references to be labeled archival
    - [x] Commit this task before starting the next task
- [x] Task: Conductor - Automated Review and Checkpoint 'Red-Phase Cutover Checks' (Protocol in workflow.md)

## Phase 2: Cutover Cleanup

- [x] Task: Update active documentation stack claims ce91fcf
    - [x] Update product status, tech-stack, and roadmap docs to agree on Starlight
    - [x] Preserve Sphinx as archival/redirect source only where needed
    - [x] Commit this task before starting the next task
- [x] Task: Resolve stale active migration tracks be948da
    - [x] Archive, merge, or remove duplicate completed Starlight track folders according to Conductor workflow
    - [x] Update `conductor/tracks.md` links if needed
    - [x] Commit this task before starting the next task
- [ ] Task: Validate Starlight route and link evidence
    - [ ] Run Starlight build and route/link generation checks
    - [ ] Record blockers for external services such as DocSearch if credentials are unavailable
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Cutover Cleanup' (Protocol in workflow.md)

## Phase 3: Final Cutover Evidence

- [ ] Task: Refresh migration manifests
    - [ ] Update cutover, redirect, and route coverage artifacts
    - [ ] Ensure all stale "canonical Sphinx" language is removed or qualified
    - [ ] Commit this task before starting the next task
- [ ] Task: Run final conductor review
    - [ ] Review the full track diff against the spec, plan, workflow, and tests
    - [ ] Apply high-confidence fixes and rerun validation
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Final Cutover Evidence' (Protocol in workflow.md)
