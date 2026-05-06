# Specification

## Overview

Turn the remaining Rust-core ownership gap into an explicit Conductor-managed
follow-on track. The current Rust roadmap already documents the mixed
ownership state, but it does not name the residual work as a concrete track.
This track makes the remainder actionable by describing the remaining
bridge-backed slices, the Python-only reference areas, and the conditions under
which future Rust ownership claims would be valid.

## Functional Requirements

1. Add an explicit follow-on track for the remaining Rust core ownership gap.
2. Update the Rust roadmap documentation so it names the remaining bridge-backed
   slices and the Rust-ownership closure track.
3. Keep the tech-stack summary aligned with the roadmap wording.
4. Add governance tests that prevent the remaining Rust gap from becoming
   implicit or getting dropped from the documentation.

## Non-Functional Requirements

1. Do not claim full Rust ownership prematurely.
2. Keep the documentation concise and audit-friendly.
3. Preserve the existing mixed Rust/Python runtime semantics.

## Acceptance Criteria

1. The Rust roadmap explicitly names the remaining ownership gap track.
2. The tech-stack documentation points to the same remaining-gap wording.
3. Tests cover the new track linkage and the remaining-gap narrative.
4. The Conductor registry records the track and its completion state cleanly.

## Out of Scope

1. Rewriting every remaining kernel operation into Rust.
2. Changing the public API or dispatch contract.
3. Removing Python bridge fallback behavior.
