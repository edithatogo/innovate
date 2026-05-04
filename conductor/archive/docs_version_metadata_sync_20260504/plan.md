# Implementation Plan: Documentation Version Metadata Synchronization

## Phase 1: Version Sync Guard

- [x] Task: Add failing documentation versioning test
    - [x] Assert Sphinx release/version match package metadata.
    - [x] Assert stale hard-coded `1.0.0` values are absent.
- [x] Task: Source Sphinx version from package metadata
    - [x] Replace hard-coded release/version values in `docs/source/conf.py`.
- [x] Task: Validate docs versioning
    - [x] Run focused docs-version tests and docs smoke build.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Version Sync Guard' (Protocol in workflow.md)
