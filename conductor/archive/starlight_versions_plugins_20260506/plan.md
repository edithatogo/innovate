# Implementation Plan

## Phase 1: Confirm the Starlight baseline

- [x] Task: Review the official Starlight package and plugin references
    - [x] Confirm the current target `@astrojs/starlight` version
    - [x] Confirm the current versions of `starlight-versions`, `starlight-links-validator`, and `@astrojs/starlight-docsearch`
    - [x] Capture which plugins are required versus optional
    - [x] Record whether DocSearch is selected or left as a future option
- [x] Task: Conductor - Automated Review and Checkpoint 'Confirm the Starlight baseline' (Protocol in workflow.md)

## Phase 2: Update the roadmap and stack docs

- [x] Task: Edit the roadmap and tech-stack documentation
    - [x] Add Starlight to the docs/tooling section of the stack
    - [x] Record the selected Starlight version and plugin set
    - [x] Explain the purpose of versioned docs, link validation, and search support
    - [x] Keep the wording consistent with the existing roadmap style
- [x] Task: Conductor - Automated Review and Checkpoint 'Update the roadmap and stack docs' (Protocol in workflow.md)

## Phase 3: Add validation for the documented choices

- [x] Task: Add or update documentation tests
    - [x] Verify the roadmap mentions the Starlight versioning policy
    - [x] Verify the approved plugins remain listed in the docs
    - [x] Verify the docs distinguish required and optional plugin choices
- [x] Task: Run the relevant doc and unit test suite
    - [x] Confirm the repo stays lint-clean for the touched files
    - [x] Confirm the track can be reviewed without manual cleanup
- [x] Task: Conductor - Automated Review and Checkpoint 'Add validation for the documented choices' (Protocol in workflow.md)

## Phase 4: Close out the track

- [x] Task: Final review and archive readiness
    - [x] Confirm the roadmap update is complete
    - [x] Confirm the track metadata reflects completion
    - [x] Confirm the worktree is clean before archive
- [x] Task: Conductor - Automated Review and Checkpoint 'Close out the track' (Protocol in workflow.md)
