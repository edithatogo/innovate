# Implementation Plan: Starlight Cutover and Legacy Cleanup

## Phase 1: Red-Phase Cutover Checks

- [ ] Task: Inventory active and legacy docs surfaces
    - [ ] Compare product, tech-stack, Starlight config, Sphinx config, and migration manifests
    - [ ] Identify stale migration-in-progress language
    - [ ] Commit this task before starting the next task
- [ ] Task: Write failing cutover tests
    - [ ] Reject product docs claiming Sphinx is the active docs stack
    - [ ] Reject active track folders for completed Starlight migration work
    - [ ] Require legacy Sphinx references to be labeled archival
    - [ ] Commit this task before starting the next task
- [ ] Task: Conductor - Automated Review and Checkpoint 'Red-Phase Cutover Checks' (Protocol in workflow.md)

## Phase 2: Cutover Cleanup

- [ ] Task: Update active documentation stack claims
    - [ ] Update product status, tech-stack, and roadmap docs to agree on Starlight
    - [ ] Preserve Sphinx as archival/redirect source only where needed
    - [ ] Commit this task before starting the next task
- [ ] Task: Resolve stale active migration tracks
    - [ ] Archive, merge, or remove duplicate completed Starlight track folders according to Conductor workflow
    - [ ] Update `conductor/tracks.md` links if needed
    - [ ] Commit this task before starting the next task
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
