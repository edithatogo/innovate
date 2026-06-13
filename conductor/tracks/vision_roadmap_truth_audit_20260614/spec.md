# Specification: Vision and Roadmap Truth Audit

## Overview

The Conductor registry marks all existing roadmap tracks complete, but the product
vision and roadmap documents still contain mixed signals: some pages describe the
roadmap as complete, while others correctly state that full Rust ownership,
external submissions, and documentation cutover remain future-state work. This
track reconciles those claims without overstating maturity.

## Functional Requirements

1. Audit product, roadmap, architecture, release, registry, HPC, and docs-site
   pages for completion language.
2. Produce a single canonical vision-status statement that distinguishes:
   completed Conductor tracks, implemented product capabilities, documented
   future-state boundaries, blocked external actions, and intentionally deferred
   work.
3. Update docs so every roadmap-level claim links to evidence or a follow-on
   track.
4. Add regression tests that fail when docs claim full completion while known
   future-state gaps remain.
5. Register granular follow-on tracks for any uncovered roadmap or vision gap.

## Non-Functional Requirements

1. Preserve the project rule that no claim of external submission, full Rust
   ownership, or completed cutover may exist without evidence.
2. Keep user-facing wording precise and non-marketing-oriented.
3. Do not remove legacy evidence unless a cutover track explicitly owns removal.

## Acceptance Criteria

1. Product and roadmap docs agree on whether the vision is complete, partially
   complete, or future-state.
2. The roadmap coverage map includes explicit entries for remaining future-state
   tracks.
3. Tests cover the canonical status wording and reject stale Sphinx/Rust
   completion claims.
4. Every task has a commit, and every phase and the final track receive
   `conductor-review`.

## Out of Scope

1. Implementing Rust-native operation coverage.
2. Executing external registry submissions.
3. Removing legacy Sphinx artifacts.
