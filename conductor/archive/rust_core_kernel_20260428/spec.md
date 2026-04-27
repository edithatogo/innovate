# Specification: Rust Core Kernel Roadmap and C# Binding Foundation

## Overview

Create the next architecture track for evolving `innovate` from a Python-first library with thin bindings into a contract-first platform with a Rust-backed core trajectory. This track turns ADR 0004 into implementation-ready work by defining Rust parity expectations, identifying the first kernel operations suitable for Rust implementation, and preparing C# as the next planned thin binding surface.

## Functional Requirements

1. Define a Rust core migration roadmap that preserves the canonical Python API and existing functional kernel contract.
2. Identify the first stable kernel operations that are suitable for Rust-backed execution behind the existing schemas.
3. Add parity-test expectations between Python reference semantics and future Rust-backed execution.
4. Define the C# binding scope, package layout, invocation path, and schema-compatibility expectations.
5. Update project documentation so Rust is clearly treated as the strategic core runtime direction while C# is treated as a planned thin binding.

## Non-Functional Requirements

1. Rust migration work must not fork model semantics away from the Python reference implementation.
2. New binding or core plans must remain schema-compatible with the functional kernel contract.
3. Documentation must distinguish current implemented behavior from planned runtime evolution.
4. The track must avoid introducing a second public API before the canonical API and kernel contract are stable.

## Acceptance Criteria

1. A Rust core roadmap exists with concrete first operations, parity-test requirements, and benchmark gates.
2. A C# binding plan exists with package structure, bridge strategy, and schema-compatibility requirements.
3. Documentation and Conductor context describe Python as the ergonomic reference surface, bindings as thin contract surfaces, and Rust as the long-term core runtime.
4. Tests or validation checks exist to prevent future bindings from drifting away from the kernel schema contract.

## Out of Scope

1. Rewriting the full kernel in Rust in a single track.
2. Publishing language packages to external registries.
3. Replacing the canonical Python public API.
4. Implementing every model family in Rust before parity and benchmark gates are defined.
