# Specification: Rust-Native Canonical Operation Completion

## Overview

The Rust roadmap states that the core is not fully Rust-owned and that full Rust
ownership cannot be claimed until every canonical operation has native or
explicitly promoted ownership. This track addresses operation-level gaps for
`discover_models`, `fit_model`, `predict_model`, `simulate_model`,
`summarize_model`, and `diagnose_model`.

## Functional Requirements

1. Read `docs/source/_static/rust_core_migration_inventory.json` as the source
   of truth for operation ownership.
2. For each canonical operation, identify all slices that are `python_bridge`,
   `python_reference`, or `native_candidate_needs_evidence`.
3. Implement or promote Rust-native operation slices where schemas are stable.
4. Preserve explicit unsupported errors for payloads that cannot safely become
   Rust-native.
5. Add parity, schema, error mapping, benchmark, memory, and binding smoke
   evidence for every promoted slice.
6. Update the inventory, Rust roadmap, and tests after each operation slice.

## Non-Functional Requirements

1. Do not bypass the functional kernel contract.
2. Do not duplicate Python model logic in bindings outside the Rust core.
3. Keep Python reference semantics authoritative until parity passes.
4. Commit after every task and run `conductor-review` after every phase and at
   final track completion.

## Acceptance Criteria

1. No canonical operation remains bridge-owned unless it is explicitly promoted
   to a non-Python backend or documented as an intentional Python boundary.
2. Each native operation slice has passing parity and schema compatibility tests.
3. Rust benchmark and memory evidence is present or explicitly marked
   not-applicable with rationale.
4. Binding smoke tests pass for the promoted operation surface.

## Out of Scope

1. Migrating model families whose payload schemas are not stable.
2. Claiming full Rust ownership for Python-only probabilistic runtimes.
3. External package registry publication.
