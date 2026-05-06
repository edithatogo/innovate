# Specification: Rust Bass Promotion Evidence and Benchmark Coverage

## Overview

Add Bass-native benchmark coverage and the supporting promotion evidence needed
to judge whether the Rust Bass slice is ready for broader promotion. The track
keeps the work focused on the currently implemented Bass native slice rather
than claiming full Rust ownership of the core.

## Functional Requirements

1. Add Criterion benchmark coverage for the Rust-native Bass predict and
   simulate slices.
2. Keep the existing logistic benchmark coverage intact.
3. Record the Bass benchmark evidence in the Rust promotion dossier and the
   Rust migration inventory.
4. Keep the Rust roadmap honest about the current state of Rust ownership.
5. Preserve the current Python bridge fallback behavior for unsupported Bass
   shapes.

## Non-Functional Requirements

1. The benchmark harness must stay reproducible with the existing Rust crate
   tooling.
2. The new evidence path must not change public kernel semantics.
3. The new records must remain machine-readable and easy to validate in CI.

## Acceptance Criteria

1. The Rust benchmark harness includes Bass-native predict and simulate cases.
2. The migration inventory explicitly calls out Bass benchmark and profiling
   evidence requirements.
3. The roadmap and dossier remain aligned with the current mixed Rust/Python
   ownership model.
4. The track is represented in the Conductor registry.

## Out of Scope

1. Claiming the core is entirely Rust.
2. Adding new Bass model families or changing Bass semantics.
3. Collecting final release-candidate benchmark measurements in this track.
