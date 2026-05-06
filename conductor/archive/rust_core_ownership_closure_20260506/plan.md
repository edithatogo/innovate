# Implementation Plan

## Phase 1: Define the remaining Rust ownership gap

- [x] Task: Specify the residual Rust-core scope
    - [x] Enumerate the remaining bridge-backed slices
    - [x] Enumerate the Python-only reference slices and payloads
    - [x] State why full Rust ownership is still not claimed
- [x] Task: Conductor - Automated Review and Checkpoint 'Define the remaining Rust ownership gap' (Protocol in workflow.md)

## Phase 2: Update the roadmap and stack docs

- [x] Task: Update the Rust roadmap narrative
    - [x] Name the ownership-closure follow-on track explicitly
    - [x] Keep the audited status aligned with the current mixed runtime
    - [x] Preserve the operation-by-operation promotion language
- [x] Task: Update the tech-stack summary
    - [x] Reflect the same remaining-gap wording
    - [x] Keep the Rust strategy statement consistent with the roadmap
- [x] Task: Conductor - Automated Review and Checkpoint 'Update the roadmap and stack docs' (Protocol in workflow.md)

## Phase 3: Add governance coverage

- [x] Task: Extend documentation tests
    - [x] Verify the roadmap names the new follow-on track
    - [x] Verify the remaining-gap wording is still present
    - [x] Verify the tech-stack wording stays aligned
- [x] Task: Run focused validation
    - [x] Check the updated docs and tests
    - [x] Confirm the track registry and archive layout are clean
- [x] Task: Conductor - Automated Review and Checkpoint 'Add governance coverage' (Protocol in workflow.md)

## Phase 4: Close out the track

- [x] Task: Final review and archive readiness
    - [x] Confirm the documentation now makes the residual Rust gap explicit
    - [x] Confirm validation passes
    - [x] Confirm the worktree is clean before archive
- [x] Task: Conductor - Automated Review and Checkpoint 'Close out the track' (Protocol in workflow.md)
