# Specification: R Bindings over the Functional Kernel

## Overview

Expose the stable `innovate` kernel to R users through a thin, well-tested binding layer that feels native to R while remaining contract-compatible with the shared kernel semantics.

## Functional Requirements

1. Define how R code invokes the functional kernel and exchanges structured data.
2. Provide R-facing wrappers for model discovery, fit, predict, simulate, and diagnostics for stable model families.
3. Map kernel schemas into idiomatic R data structures and error handling.
4. Add package metadata, examples, and tests for the R binding layer.
5. Document installation and backend requirements clearly for R users.

## Non-Functional Requirements

1. The R layer must be thin and avoid duplicating model logic.
2. Schema compatibility with the kernel must be tested automatically.
3. The binding should be designed for CRAN-aware packaging constraints where practical.

## Acceptance Criteria

1. An R package skeleton or equivalent binding structure exists in the repository.
2. Stable kernel operations are accessible from R with tests.
3. Data conversion and error mapping are documented and validated.
4. Users can run a simple end-to-end example from R against the stable kernel.

## Out of Scope

1. CRAN publication.
2. Wrapping every experimental model family.
3. A separate R-native modeling implementation.
