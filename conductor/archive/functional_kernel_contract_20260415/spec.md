# Specification: Functional Kernel Contract

## Overview

Create a language-neutral functional core for stable model execution so `innovate` can expose the same semantics to Python, R, Julia, and future bindings without coupling those ecosystems to Python class internals, NumPy-only semantics, or accelerator-specific artifacts.

## Functional Requirements

1. Define canonical kernel functions for model discovery, fit, predict, simulate, summarize, and diagnostics.
2. Define request and response schemas that use stable, serializable, Arrow-compatible data structures where tabular payloads are involved.
3. Implement kernel adapters for the stable core model families.
4. Add schema validation and versioning rules so the kernel contract can evolve safely.
5. Provide clear error semantics that are usable from non-Python bindings.

## Non-Functional Requirements

1. The kernel must avoid reliance on Python-specific object identity or inheritance in its public contract.
2. Numerical semantics should be Array API-friendly where practical so the contract is not tied to one array implementation.
3. Schemas must be versioned and backward-compatible where practical.
4. The kernel must compose with the canonical public API rather than fork behavior.

## Acceptance Criteria

1. A documented kernel contract exists with versioned request and response shapes.
2. Core stable model families can be executed through the kernel.
3. Tests cover round-tripping of kernel inputs and outputs.
4. Documentation explains how the kernel relates to the Python OO API, Arrow interchange, and future bindings.

## Out of Scope

1. Completing every possible model family.
2. Network transport or hosted service deployment.
3. Full implementation of all downstream language bindings.
