# Implementation Plan

## Phase 1: Define the local version-sync session

- [x] Task: Specify the `nox` session behavior
    - [x] Default to check mode
    - [x] Support write mode through session arguments
    - [x] Align the session with `uv`-first Python tooling
- [x] Task: Conductor - Automated Review and Checkpoint 'Define the local version-sync session' (Protocol in workflow.md)

## Phase 2: Implement the session and docs

- [x] Task: Add the `version_sync` nox session
    - [x] Invoke `scripts/sync_versions.py`
    - [x] Forward check/write arguments to the script
    - [x] Keep the session discoverable in `nox --list`
- [x] Task: Update maintainer documentation
    - [x] Explain the local release-prep workflow
    - [x] Point maintainers to the new nox session
- [x] Task: Conductor - Automated Review and Checkpoint 'Implement the session and docs' (Protocol in workflow.md)

## Phase 3: Add validation

- [x] Task: Extend governance tests
    - [x] Verify the nox session is advertised
    - [x] Verify the docs mention the local version-sync workflow
- [x] Task: Run focused validation
    - [x] Check the sync script and nox session behavior
    - [x] Confirm docs and tests stay clean
- [x] Task: Conductor - Automated Review and Checkpoint 'Add validation' (Protocol in workflow.md)

## Phase 4: Close out the track

- [x] Task: Final review and archive readiness
    - [x] Confirm the session is documented
    - [x] Confirm the validation passes
    - [x] Confirm the worktree is clean before archive
- [x] Task: Conductor - Automated Review and Checkpoint 'Close out the track' (Protocol in workflow.md)
