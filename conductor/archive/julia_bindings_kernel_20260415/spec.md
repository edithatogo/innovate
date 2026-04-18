# Specification: Julia Bindings over the Functional Kernel

## Overview

Expose the stable `innovate` kernel to Julia users through a thin, well-tested binding layer that preserves Julia ergonomics while keeping model semantics centralized in the shared kernel contract.

## Functional Requirements

1. Define how Julia code invokes the kernel and exchanges structured data.
2. Provide Julia-facing wrappers for model discovery, fit, predict, simulate, and diagnostics for stable model families.
3. Map kernel schemas into idiomatic Julia types and exception handling.
4. Add package metadata, examples, and tests for the Julia binding layer.
5. Document installation and backend expectations clearly for Julia users.

## Non-Functional Requirements

1. The Julia layer must remain thin and contract-driven.
2. Schema compatibility with the kernel must be tested automatically.
3. The binding structure must support future registration and packaging workflows in the Julia ecosystem.

## Acceptance Criteria

1. A Julia package skeleton or equivalent binding structure exists in the repository.
2. Stable kernel operations are accessible from Julia with tests.
3. Data conversion and error mapping are documented and validated.
4. Users can run a simple end-to-end Julia example against the stable kernel.

## Out of Scope

1. Julia package registry publication.
2. Wrapping every experimental model family.
3. A separate Julia-native modeling implementation.
