# Specification: Rust Core Summary and Diagnostics Migration

## Overview

Extend the Rust-native kernel path so fitted-state `summarize_model` and `diagnose_model` requests for simple logistic models can execute without invoking the Python bridge. The new native path must preserve the existing kernel schema, diagnostics shape, and fallback behavior for unsupported model families or payloads.

## Functional Requirements

1. Add a Rust-native execution path for `summarize_model` on simple fitted logistic payloads.
2. Add a Rust-native execution path for `diagnose_model` on simple fitted logistic payloads.
3. Preserve Python bridge fallback for non-native model families and unsupported payload shapes.
4. Return diagnostics, state serialization, and metadata that remain compatible with the existing Python kernel contract.
5. Add parity tests that compare native Rust output against the Python bridge contract for both summary and diagnostics.

## Non-Functional Requirements

1. Native Rust execution must not change the public kernel schema or error codes.
2. The implementation must remain narrow and deterministic enough to support stable parity tests.
3. Documentation must distinguish the implemented native slice from the still-bridge-backed execution paths.

## Acceptance Criteria

1. `summarize_model` has a native Rust path for simple logistic fitted-state requests.
2. `diagnose_model` has a native Rust path for simple logistic fitted-state requests.
3. Unsupported payloads still fall back to the Python bridge.
4. Tests verify parity against the Python bridge contract for both operations.
5. Rust binding documentation and the Rust core roadmap are updated to mention the new native slice.

## Out of Scope

1. Rewriting the full diagnostics pipeline in Rust for every model family.
2. Changing the canonical Python API or kernel schema.
3. Packaging or publishing new language artifacts.
