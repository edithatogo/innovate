# Specification: Optional Dependency Isolation, Backend Stabilization, and Engine Positioning

## Overview

Make the base Python package robust without optional scientific extras, while explicitly supporting JAX and Bayesian backends through clear extras, capability metadata, and isolated test coverage.

## Functional Requirements

1. Ensure the base install does not require JAX, ArviZ, BlackJAX, or other optional accelerator packages to import or collect tests.
2. Define explicit extras and documented installation modes for optional backend capabilities.
3. Add a backend compatibility surface that makes it clear which models and fitters support which backends.
4. Add graceful runtime errors or warnings when users request unsupported backends or missing optional dependencies.
5. Define the project position for pandas, PyArrow, and any selective Polars usage so tabular-engine choices do not become implicit or ad hoc.
6. Separate test execution so base and optional-backend suites can run independently.

## Non-Functional Requirements

1. Base install behavior must be deterministic and easy to support.
2. Optional backend support must be discoverable without reading source code.
3. Backend handling must not create hidden import side effects.
4. XLA/JAX internals must not be treated as the public ABI for `innovate`.
5. Any segmentation-fault-prone or unstable Bayesian path must be clearly marked experimental until stabilized.

## Acceptance Criteria

1. Base environment test collection succeeds without JAX/Bayesian extras installed.
2. Optional extras are documented and exercised by dedicated tests.
3. Backend capability metadata is exposed to callers and reflected in docs.
4. Missing optional dependency paths fail with clear messages rather than import-time crashes.
5. Documentation clearly states that pandas plus PyArrow is the primary Python tabular surface and that Polars, if used, is selective and non-contractual.

## Out of Scope

1. Adding new inference methods.
2. Changing the long-term kernel contract beyond clarifying backend boundaries.
3. Creating non-Python bindings.
