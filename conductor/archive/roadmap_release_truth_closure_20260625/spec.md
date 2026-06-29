# Roadmap Release Truth Closure

## Overview

Close the gap between completed Conductor tracks and the full product vision by
building a truth ledger that maps every roadmap claim to current evidence,
active work, external blockers, or explicit future-state deferral. This track
prevents release, README, docs, and registry language from overstating maturity
before Rust ownership, polyglot bindings, Starlight docs, dependency posture,
CI, code quality, and external acceptance gates are genuinely complete.

## Functional Requirements

- Inventory every roadmap item, ADR, product vision claim, archived track, active
  track, and static release evidence artifact.
- Classify each item as complete, active, external-blocked, future-state, or
  intentionally out of scope.
- Ensure every incomplete item points to an active Conductor track.
- Update docs and release evidence so claims use the truth ledger as source of
  truth.
- Add tests that fail if roadmap claims are made without evidence or a track.
- Include related modeling claims for policy diffusion, competition,
  substitution, network diffusion, and advanced runtime surfaces.

## Non-Functional Requirements

- Claims must be fail-closed: no acceptance, maturity, Rust-native ownership, or
  registry publication claim without current evidence.
- The ledger must be machine-readable and easy to audit in CI.
- Existing archived track evidence must be preserved, not rewritten.

## Acceptance Criteria

- A machine-readable roadmap truth ledger exists and covers every roadmap item.
- Active tracks cover all incomplete roadmap items.
- Unit tests validate ledger coverage against roadmap docs and Conductor tracks.
- Release-readiness evidence references the ledger.
- `uv run nox -s lint types tests docs package` passes or any remaining failure
  is recorded as an explicit blocker.
- Phase and track completion follow Conductor review, push, and GitHub Actions
  monitoring requirements.

## Out Of Scope

- Implementing Rust-native model behavior.
- Publishing packages to external registries.
- Removing Starlight migration artifacts.
