# Specification: ABI and Binary Compatibility Strategy

## Overview

Define an API-preserving ABI and binary-compatibility strategy that supports
Rust-native and HPC deployment work without breaking the existing public API.

## Dependencies

- Feeds Rust core migration execution.
- Feeds HPC packaging and registry readiness.
- Feeds polyglot documentation architecture.

## Functional Requirements

1. Distinguish public API compatibility, kernel schema compatibility, native
   ABI compatibility, and backend capability metadata.
2. Define the Arrow C Data Interface and Arrow C Stream Interface as the
   preferred native interchange boundaries for binary-compatible tabular data.
3. Define what must not become public ABI, including XLA internals, jaxlib
   objects, HLO or compiled executable handles, Rust private structs, and Python
   object layouts.
4. Define semver, schema-version, and capability-discovery rules for native
   implementation changes.
5. Document package-manager binary compatibility notes for PyPI, conda,
   crates.io, npm, CRAN/R-universe, Julia General, Go modules, and NuGet.

## Parallelization

- Agent A owns Python and schema compatibility policy.
- Agent B owns Arrow and native interchange policy.
- Agent C owns Rust crate and FFI policy.
- Agent D owns XLA and accelerator non-ABI policy.
- Agent E owns package-manager binary compatibility notes.
- Agent F owns docs, tests, and cross-language review.

## Acceptance Criteria

1. ABI policy preserves the public API and separates API, schema, native ABI,
   and capability metadata.
2. Arrow C Data Interface and Arrow C Stream Interface are documented as the
   native binary interchange boundary.
3. XLA internals and Rust/Python private implementation details are explicitly
   rejected as public ABI.
4. Package-manager binary compatibility notes cover Python wheels, conda,
   crates.io, npm, R, Julia, Go, and NuGet.
5. Static tests enforce the ABI policy language and Conductor registry state.
