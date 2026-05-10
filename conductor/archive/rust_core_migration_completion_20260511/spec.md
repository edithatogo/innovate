# Specification: Rust Core Migration Completion and Polyglot Claim Closure

## Overview

Complete the remaining Rust-core ownership gap so the repository can honestly
describe itself as a polyglot library with a Rust-owned core for the promoted
kernel slices. The current project already has a stable functional kernel,
Arrow interchange, multiple language bindings, and a partial Rust runtime. This
track closes the remaining bridge-backed and Python-only slices, tightens the
ABI boundary, and updates the docs and evidence bundle so the migration claim
matches the implemented state.

## Background

The current repository state explicitly says the core is not yet fully
Rust-owned. The Rust roadmap still tracks bridge-backed slices, Python-only
reference areas, and remaining model families. The polyglot architecture docs
also state that the current source layout remains canonical until a dedicated
migration track justifies a move. This track turns that remaining gap into a
single execution plan.

## Functional Requirements

1. Inventory every canonical operation, stable payload shape, and registry
   model family that still depends on Python reference behavior or bridge
   fallback.
2. Promote every remaining canonical operation and stable model slice to
   Rust-native execution, or explicitly classify the surface as permanently
   non-native with a documented reason.
3. Remove bridge fallback for any slice that is claimed as Rust-owned, while
   preserving explicit fallback behavior for surfaces that remain intentionally
   bridge-backed.
4. Preserve the public kernel contract, schema compatibility rules, and
   capability-discovery semantics across Python, Rust, R, Julia, Go, TypeScript,
   and C#.
5. Tighten ABI and binary-compatibility policy so native slices do not expose
   unstable Python internals, jaxlib internals, or Rust private structs.
6. Update machine-readable migration inventory, roadmap docs, binding docs, and
   release/readiness docs so the ownership claim is reflected consistently.
7. Maintain benchmark, parity, and profiling evidence for each promoted slice.
8. Keep package metadata, smoke tests, and publication gates aligned with the
   promoted Rust-owned surfaces.

## Non-Functional Requirements

1. Changes must be deterministic and reproducible.
2. Native execution must fail clearly when a request is outside the promoted
   Rust slice rather than silently drifting to an undocumented path.
3. The migration must not break existing binding APIs or community-facing docs.
4. Evidence must remain auditable in docs, static fixtures, and test output.
5. Any source-tree move remains out of scope unless a separate track proves it
   is necessary.

## Acceptance Criteria

1. The Rust core roadmap no longer describes unresolved bridge-backed gaps for
   any canonical operation or stable payload shape that is claimed as Rust
   owned.
2. Every remaining model family is either native Rust, explicitly promoted to a
   non-Python backend, or documented as intentionally non-native.
3. Bridge fallback remains only for surfaces that are intentionally not
   promoted.
4. The machine-readable migration inventory reflects the final ownership state
   and is consumed by tests.
5. ABI and capability-discovery tests enforce the new ownership boundary.
6. Binding tests for R, Julia, Go, TypeScript, Rust, and Python continue to
   pass against the promoted core.
7. The roadmap, architecture, and readiness docs all describe the final state
   consistently.
8. The full validation suite passes, including docs build, binding tests,
   parity tests, and review checkpointing.

## Out of Scope

1. Unrelated model research or new scientific features.
2. New language bindings.
3. A source-tree relocation that is not already justified by a separate
   migration track.
4. Community submissions that depend on this migration but are not part of the
   Rust-core closure itself.

