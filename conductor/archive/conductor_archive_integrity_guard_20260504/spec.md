# Specification: Conductor Archive Metadata and Decision-Link Integrity Guard

## Overview

The Conductor registry marks all archived roadmap and implementation tracks as
completed, but several archived `metadata.json` files still report `new`. ADR
0005 is also present in the ADR index but not covered by the roadmap decision
link tests. This track makes status reporting auditable and prevents future
registry/archive drift.

## Functional Requirements

- Normalize archived track metadata so every track linked from
  `conductor/tracks.md` as completed has `metadata.status` set to `completed`.
- Add a registry-wide unit test that parses completed archive links from
  `conductor/tracks.md` and verifies each archive directory contains complete
  Conductor artifacts and completed metadata.
- Add ADR 0005 to roadmap decision-link coverage.
- Keep the change scoped to Conductor governance, roadmap documentation, and
  tests.

## Acceptance Criteria

- All completed archive links in `conductor/tracks.md` resolve to archive
  directories with `metadata.json`, `spec.md`, `plan.md`, and `index.md`.
- Each linked archive metadata file has `status` equal to `completed`.
- ADR 0005 is listed in `docs/architecture_modernization_roadmap.md` decision
  links and covered by `tests/unit/test_roadmap_backlog_tracks.py`.
- Focused roadmap tests pass locally.

## Out of Scope

- Runtime API behavior changes.
- New package-publication workflows.
- Process-mining fixture implementation.
