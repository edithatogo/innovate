# Specification: Rust Core Full Native Migration and Ownership Closure

## Overview

Complete the remaining Rust-core migration so that every canonical operation,
every Python registry model family, and every stable payload shape is either
native in Rust or explicitly promoted elsewhere with a documented owner and
fallback policy. The repository already contains the current Rust-native
slices, the migration inventory, and the roadmap; this track closes the
remaining ownership gaps, maintains a machine-readable slice ledger, and
updates the claim language only after the evidence supports it.

## Background

The Rust roadmap already states that the core is not fully Rust-owned today and
that the remaining gap is tracked in Conductor. This track is the follow-on
work needed to close that gap without changing the public product contract or
source layout.

## Functional Requirements

1. Inventory every remaining canonical operation, Python registry model
   family, and stable payload shape that is not yet Rust-native.
2. Maintain a machine-readable slice ledger that records every tracked slice
   with a terminal ownership state and evidence pointers.
3. Classify each remaining slice as one of:
   - Rust-native
   - explicitly promoted to a non-Rust owner
   - explicitly remaining Python-reference-owned
4. Implement Rust-native execution for every slice that remains promotable.
5. Remove undocumented bridge fallback for promoted slices and preserve bridge
   fallback only where the ownership policy explicitly allows it.
6. Preserve parity, error mapping, benchmark, profiling, and binding smoke
   evidence for every promoted slice.
7. Update the Rust roadmap, migration inventory, and related docs so they state
   the same ownership picture.
8. Keep language bindings aligned with the final ownership state across
   Python, R, Julia, TypeScript, Go, and C#.
9. Reconcile claim language in the roadmap and adjacent docs so no page says
   "partial" or equivalent wording once the ledger says otherwise.

## Non-Functional Requirements

1. The migration must remain reproducible from repo artifacts and tests.
2. The track must not claim full Rust ownership without auditable evidence.
3. The source tree must remain stable unless a separate migration track says
   otherwise.
4. The track must not introduce unrelated model research.
5. The slice ledger and the docs inventory must remain synchronized.

## Acceptance Criteria

1. Every canonical operation has a terminal ownership state recorded in the
   inventory.
2. Every Python registry model family is either Rust-native or explicitly
   promoted elsewhere with rationale.
3. Every stable payload shape is either native or explicitly owned by a
   non-Rust backend.
4. The machine-readable slice ledger exists and matches the rendered docs.
5. Promoted slices have parity, error mapping, profiling, benchmark, and
   binding-smoke evidence.
6. Each promoted family has a binding smoke matrix covering the applicable
   Rust, Python, and downstream language surfaces.
7. Each promoted family has benchmark and profiling evidence attached to the
   ledger or the supporting evidence bundle.
8. The Rust roadmap no longer overstates bridge-backed ownership.
9. The track can be archived cleanly once the ownership boundary is explicit.

## Out of Scope

1. New source-tree relocations.
2. New language bindings.
3. Model research unrelated to the ownership boundary.
4. Docs-site migration work.
