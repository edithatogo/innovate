# Bleeding-Edge Dependency Modernization

## Overview

Keep the repo aligned with the selected bleeding-edge baseline: Python 3.14,
NumPy 2+, Polars-preferred dataframes, Pydantic v2, basedpyright strict, Astro
7/Starlight, TypeScript 6, Node 26 types, Vitest 4, criterion 0.8, current
mutmut, and ecosystem-specific dashboards.

## Functional Requirements

- Audit Python, Node, Rust, R, Julia, Go, and .NET dependency manifests.
- Align minimum/runtime floors with validated CI lanes.
- Replace pandas usage with Polars where product behavior allows.
- Ensure dependency dashboards report outdated packages for all ecosystems.
- Add tests/evidence that Python 3.14 is the required default lane.

## Non-Functional Requirements

- Avoid unbounded upgrades where scientific APIs are unstable; document upper
  bounds when needed.
- Lockfiles must be reproducible.
- CI should fail on dependency drift or unsupported runtime floors.

## Acceptance Criteria

- Dependency dashboards run locally and in CI.
- Runtime floors match CI.
- basedpyright strict targets Python 3.14.
- Package metadata and docs agree on dependency baselines.
- Full nox, docs, and package checks pass or blockers are explicit.

## Out Of Scope

- Publishing packages.
- Rewriting unrelated model algorithms.
