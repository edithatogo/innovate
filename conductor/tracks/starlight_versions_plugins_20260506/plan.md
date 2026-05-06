# Implementation Plan

## Phase 1: Confirm the Starlight baseline

- [ ] Task: Review the official Starlight package and plugin references
    - [ ] Confirm the current target `@astrojs/starlight` version
    - [ ] Confirm the current versions of the approved plugins
    - [ ] Capture which plugins are required versus optional
- [ ] Task: Conductor - Automated Review and Checkpoint 'Confirm the Starlight baseline' (Protocol in workflow.md)

## Phase 2: Update the roadmap and stack docs

- [ ] Task: Edit the roadmap and tech-stack documentation
    - [ ] Add Starlight to the docs/tooling section of the stack
    - [ ] Record the selected Starlight version and plugin set
    - [ ] Explain the purpose of versioned docs, link validation, and search support
    - [ ] Keep the wording consistent with the existing roadmap style
- [ ] Task: Conductor - Automated Review and Checkpoint 'Update the roadmap and stack docs' (Protocol in workflow.md)

## Phase 3: Add validation for the documented choices

- [ ] Task: Add or update documentation tests
    - [ ] Verify the roadmap mentions the Starlight versioning policy
    - [ ] Verify the approved plugins remain listed in the docs
    - [ ] Verify the docs distinguish required and optional plugin choices
- [ ] Task: Run the relevant doc and unit test suite
    - [ ] Confirm the repo stays lint-clean for the touched files
    - [ ] Confirm the track can be reviewed without manual cleanup
- [ ] Task: Conductor - Automated Review and Checkpoint 'Add validation for the documented choices' (Protocol in workflow.md)

## Phase 4: Close out the track

- [ ] Task: Final review and archive readiness
    - [ ] Confirm the roadmap update is complete
    - [ ] Confirm the track metadata reflects completion
    - [ ] Confirm the worktree is clean before archive
- [ ] Task: Conductor - Automated Review and Checkpoint 'Close out the track' (Protocol in workflow.md)
