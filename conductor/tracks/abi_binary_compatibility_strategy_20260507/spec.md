# Specification

## Overview

Define an ABI and binary-compatibility strategy that supports Rust-native and
HPC deployment work without breaking the existing public API.

## Dependencies

- Feeds Rust core migration execution.
- Feeds HPC packaging and registry readiness.
- Feeds polyglot documentation architecture.

## Functional Requirements

1. Distinguish public API, schema compatibility, and native ABI.
2. Define where Arrow C Data Interface compatibility is relevant.
3. Define what must not become public ABI, including XLA internals and Rust
   private structs.
4. Define semver, schema-version, and capability-discovery rules for native
   implementation changes.

## Parallelization

- Agent A owns Python and schema compatibility policy.
- Agent B owns Arrow and native interchange policy.
- Agent C owns Rust crate and FFI policy.
- Agent D owns XLA and accelerator non-ABI policy.
- Agent E owns package-manager binary compatibility notes.
- Agent F owns docs, tests, and cross-language review.

## Acceptance Criteria

1. ABI policy preserves the public API.
2. Native implementation details remain capability-gated.
3. Package and HPC tracks have enough ABI guidance to proceed.
