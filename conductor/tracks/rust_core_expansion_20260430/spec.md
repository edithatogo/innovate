# Specification: Rust Core Expansion

## Overview

Expand Rust-backed core execution only where operation-level parity, error mapping, schema compatibility, and benchmark gates justify promotion. This track turns the roadmap item "broad Rust rewrites before operation-level parity and benchmark gates exist" into a controlled follow-on track.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- Deferred work item: "broad Rust rewrites before operation-level parity and benchmark gates exist"

## Functional Requirements

1. Inventory currently Rust-backed operations, Python bridge fallbacks, and remaining Python reference operations.
2. Select the next Rust-core candidate operations based on stability, schema readiness, and benchmark value.
3. Add parity tests against Python reference semantics before enabling any Rust-backed behavior.
4. Add structured error mapping tests and bridge fallback tests.
5. Require benchmark and profiling evidence before promoting a Rust path to default execution.
6. Compare native Rust candidates against NumPy/SciPy reference behavior and eligible JAX/XLA-backed paths before default promotion.
7. Document operation-level support status and promotion criteria.

## Non-Functional Requirements

1. The canonical Python API must remain unchanged.
2. Rust-backed behavior must not fork semantics from the Python reference implementation.
3. The bridge must preserve schema compatibility for every supported binding.
4. Performance gates must include regression protections and clear fallback behavior.
5. Rust promotion gates must account for XLA compile cost, steady-state execution, portability, and deployment complexity where JAX is a viable candidate.

## Acceptance Criteria

1. Candidate Rust-core operations are documented with promotion gates.
2. At least one additional Rust-core implementation slice has parity, error mapping, fallback, and benchmark coverage.
3. Documentation identifies which operations are native, bridged, experimental, or Python-backed.
4. CI validates parity and schema compatibility for the implemented slice.
5. Promotion evidence explains why Rust-native execution is preferable to or complementary with an XLA-backed implementation for the selected operation.

## Out of Scope

1. Rewriting the full Python implementation in Rust in one pass.
2. Creating a second public API around Rust internals.
3. Removing Python reference semantics before parity and benchmark gates pass.
