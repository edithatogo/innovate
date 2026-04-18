# Specification: Go Bindings over the Functional Kernel

## Overview

Expose the stable `innovate` kernel to Go users through a thin, well-tested binding layer that follows idiomatic Go package design while keeping model semantics centralized in the shared kernel contract.

## Functional Requirements

1. Define how Go code invokes the functional kernel and exchanges structured data.
2. Provide Go-facing wrappers for model discovery, fit, predict, simulate, and diagnostics for stable model families.
3. Map kernel schemas into idiomatic Go structs, interfaces where necessary, and explicit error handling.
4. Add module metadata, examples, and tests for the Go binding layer.
5. Document installation, runtime, and backend expectations clearly for Go users.

## Non-Functional Requirements

1. The Go layer must remain thin and contract-driven.
2. Schema compatibility with the kernel must be tested automatically.
3. The binding structure must support standard Go module workflows and reproducible builds.

## Acceptance Criteria

1. A Go module skeleton or equivalent binding structure exists in the repository.
2. Stable kernel operations are accessible from Go with tests.
3. Data conversion and error mapping are documented and validated.
4. Users can run a simple end-to-end Go example against the stable kernel.

## Out of Scope

1. Publishing a standalone Go module to an external registry.
2. Wrapping every experimental model family.
3. A separate Go-native modeling implementation.
