# Specification: Rust-Native Payload and Model-Family Coverage

## Overview

Operation-level Rust support is not enough to claim full Rust ownership. The
Rust roadmap requires coverage across every Python registry model family and
every stable payload shape, or an explicit promotion to another non-Python
backend. This track owns model-family and payload-shape expansion.

## Functional Requirements

1. Inventory Python registry model families and compare them against Rust-native
   coverage.
2. Inventory stable payload shapes, including covariates, event splits,
   incomplete fitted states, diagnostics payloads, and simulation payloads.
3. Implement Rust-native coverage for stable model-family payloads in small,
   testable slices.
4. Preserve bridge fallback only for explicitly non-native families and unstable
   payloads.
5. Update docs, schemas, and migration inventory after each promoted family or
   payload shape.

## Non-Functional Requirements

1. Prefer small model-family tracks or subtasks over broad rewrites.
2. Do not promote a model family until parity, schema compatibility, error
   mapping, and smoke evidence pass.
3. Commit after every task and run `conductor-review` after every phase and full
   track completion.

## Acceptance Criteria

1. Every Python registry model family has an explicit Rust-native, promoted
   alternative, bridge-backed, or Python-reference status.
2. Every stable payload shape has a schema fixture and ownership status.
3. Tests prevent future model families from being added without ownership
   classification.
4. The Rust roadmap can accurately state whether full Rust ownership is achieved
   or which remaining families are intentionally excluded.

## Out of Scope

1. Operation-level implementation already covered by the companion operation
   completion track.
2. Probabilistic runtimes without stable schemas.
3. UI or documentation-site cutover work.
