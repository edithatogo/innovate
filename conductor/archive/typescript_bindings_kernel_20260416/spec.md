# Specification: TypeScript Bindings over the Functional Kernel

## Overview

Expose the stable `innovate` kernel to TypeScript users through a thin, well-tested binding layer that feels native in modern JavaScript and TypeScript environments while keeping model semantics centralized in the shared kernel contract.

## Functional Requirements

1. Define how TypeScript code invokes the functional kernel and exchanges structured data.
2. Provide TypeScript-facing wrappers for model discovery, fit, predict, simulate, and diagnostics for stable model families.
3. Map kernel schemas into idiomatic TypeScript types, runtime validation, and error handling.
4. Add package metadata, examples, and tests for the TypeScript binding layer.
5. Document installation, runtime, and backend expectations clearly for TypeScript users.

## Non-Functional Requirements

1. The TypeScript layer must remain thin and avoid duplicating model logic.
2. Schema compatibility with the kernel must be tested automatically.
3. The binding should support modern Node.js and package-manager workflows without locking the project into unnecessary frontend-specific assumptions.

## Acceptance Criteria

1. A TypeScript package skeleton or equivalent binding structure exists in the repository.
2. Stable kernel operations are accessible from TypeScript with tests.
3. Data conversion, runtime validation, and error mapping are documented and validated.
4. Users can run a simple end-to-end TypeScript example against the stable kernel.

## Out of Scope

1. npm publication.
2. Browser-native execution of the full modeling stack.
3. Wrapping every experimental model family.
4. A separate TypeScript-native modeling implementation.
