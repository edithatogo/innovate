# Implementation Plan: Conductor Archive Metadata and Decision-Link Integrity Guard

## Phase 1: Archive Integrity Guard

- [x] Task: Add failing archive metadata integrity coverage
    - [x] Parse completed archive links from `conductor/tracks.md`.
    - [x] Assert complete Conductor artifacts and completed metadata for every archive link.
- [x] Task: Normalize archived metadata statuses
    - [x] Update stale archived `metadata.json` files from `new` to `completed`.
- [x] Task: Add ADR 0005 decision-link coverage
    - [x] Update roadmap decision links.
    - [x] Update roadmap ADR test records.
- [x] Task: Validate focused roadmap tests
    - [x] Run `uv run pytest tests/unit/test_roadmap_backlog_tracks.py tests/unit/test_roadmap_gap_tracks.py -q`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Archive Integrity Guard' (Protocol in workflow.md)
