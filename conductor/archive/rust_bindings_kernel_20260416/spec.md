# Specification: Rust Bindings over the Functional Kernel

## Overview

Expose the stable `innovate` kernel to Rust users through a thin, well-tested binding layer that preserves Rust ergonomics and safety conventions while keeping model semantics centralized in the shared kernel contract.

## Functional Requirements

1. Define how Rust code invokes the functional kernel and exchanges structured data.
2. Provide Rust-facing wrappers for model discovery, fit, predict, simulate, and diagnostics for stable model families.
3. Map kernel schemas into idiomatic Rust types, validation, and explicit error handling.
4. Add crate metadata, examples, and tests for the Rust binding layer.
5. Document installation, runtime, and backend expectations clearly for Rust users.

## Non-Functional Requirements

1. The Rust layer must remain thin and contract-driven.
2. Schema compatibility with the kernel must be tested automatically.
3. The binding structure must support standard Cargo workflows and a path to future FFI or SDK hardening if required.

## Acceptance Criteria

1. A Rust crate skeleton or equivalent binding structure exists in the repository.
2. Stable kernel operations are accessible from Rust with tests.
3. Data conversion and error mapping are documented and validated.
4. Users can run a simple end-to-end Rust example against the stable kernel.

## Out of Scope

1. crates.io publication.
2. Wrapping every experimental model family.
3. A separate Rust-native modeling implementation.
