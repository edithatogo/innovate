# Specification: Starlight Cutover and Legacy Cleanup

## Overview

The repository contains an Astro/Starlight site and completed Starlight tracks,
but product and tech-stack docs still contain stale Sphinx migration language,
and two completed migration folders remain under `conductor/tracks/`. This track
finishes documentation cutover hygiene without deleting archival evidence.

## Functional Requirements

1. Make Astro/Starlight the only active documentation stack in product and tech
   stack status docs.
2. Keep Sphinx sources only as explicitly labeled legacy/archive references.
3. Resolve duplicate or stale live track folders for Starlight migration.
4. Verify redirects, route coverage, link validation, search decision, versioning,
   and deployment workflow status.
5. Add tests that reject stale Sphinx-as-active or migration-in-progress wording.

## Non-Functional Requirements

1. Preserve all audit evidence needed by archived tracks.
2. Do not remove docs/source until a test-covered archival policy is in place.
3. Commit after every task and run `conductor-review` after every phase and full
   track completion.

## Acceptance Criteria

1. Product status and tech stack agree that Astro/Starlight is active.
2. Legacy Sphinx content is clearly archival or compatibility material.
3. No completed Starlight migration folder remains as an active track.
4. Starlight build, route, and link checks pass or have explicit blocker notes.

## Out of Scope

1. Rewriting all documentation content.
2. Implementing Rust-core behavior.
3. External registry submissions.
