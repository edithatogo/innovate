# Specification: Rust-Native Discovery Metadata

## Overview

Implement the first Rust-backed kernel capability by adding a Rust-native `discover_models` metadata path that mirrors the Python functional kernel contract. This is a low-risk step toward the Rust core trajectory because discovery is schema-driven metadata and does not execute model fitting or prediction logic.

## Functional Requirements

1. Add a Rust-native discovery function that returns `KernelDiscoveryResponse`.
2. Keep the existing Python bridge path available for parity and operations that still require Python execution.
3. Add parity tests comparing Rust-native discovery metadata with Python bridge discovery metadata.
4. Preserve the existing public Rust binding API unless a new explicit native method is needed.
5. Document that only discovery metadata is Rust-native; model execution remains Python-backed.

## Non-Functional Requirements

1. The Rust-native response must use the same `KERNEL_SCHEMA_VERSION` as the Python kernel.
2. The Rust-native metadata must stay schema-compatible with `KernelDiscoveryRecord`.
3. The implementation must not reimplement model behavior.
4. Future registry drift must be caught by tests.

## Acceptance Criteria

1. Rust tests prove native discovery and bridge discovery return the same model keys and metadata for the supported discovery fields.
2. Existing Rust end-to-end bridge tests still pass.
3. Documentation distinguishes the native discovery metadata path from Python-backed model execution.

## Out of Scope

1. Rust-native fitting, prediction, simulation, summary, or diagnostics.
2. Replacing the Python bridge for non-discovery operations.
3. Publishing the Rust crate.
