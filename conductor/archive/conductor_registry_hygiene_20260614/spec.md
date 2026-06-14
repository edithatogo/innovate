# Specification: Conductor Registry Hygiene

## Overview

The tracks registry shows all existing tracks complete, yet completed Starlight
track folders remain under `conductor/tracks/`. This creates ambiguity for
status tooling and future implementation. This track adds guardrails so registry
state, active folders, archived folders, and status reports remain consistent.

## Functional Requirements

1. Detect active track folders that are not registered as active tracks.
2. Detect registry entries that point to missing folders or wrong archive paths.
3. Add tests or scripts that fail on registry/folder drift.
4. Reconcile stale active folders according to Conductor workflow.
5. Update status tooling documentation so future audits distinguish active,
   archived, stale, and orphaned tracks.

## Non-Functional Requirements

1. Do not delete archived evidence.
2. Preserve completed track history and links.
3. Commit after every task and run `conductor-review` after every phase and full
   track completion.

## Acceptance Criteria

1. `conductor/tracks/` contains only registered active tracks.
2. Completed tracks live under `conductor/archive/` and are linked correctly from
   `conductor/tracks.md`.
3. A regression test fails when a stale active track folder appears.
4. Status reports accurately surface orphaned or stale track folders.

## Out of Scope

1. Implementing product roadmap items.
2. Changing the Conductor workflow semantics.
3. Deleting legacy docs evidence.
