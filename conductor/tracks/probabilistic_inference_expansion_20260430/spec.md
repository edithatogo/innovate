# Specification: Probabilistic Inference Expansion

## Overview

Expand probabilistic inference coverage after the functional kernel contract, Arrow-compatible interchange, and optional backend isolation are stable. This track turns the roadmap item "wider probabilistic inference coverage" into implementation-ready work without making probabilistic runtimes mandatory for the base package.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- Deferred work item: "wider probabilistic inference coverage"

## Functional Requirements

1. Inventory existing probabilistic, Bayesian, and simulation-backed inference surfaces.
2. Select the next model families or inference routines that should receive probabilistic coverage.
3. Define versioned request and response payloads for posterior samples, uncertainty summaries, diagnostics, and provenance.
4. Prefer XLA-aligned probabilistic engines for eligible kernels, with NumPyro as the first-class modelling path and BlackJAX as the lower-level sampler path when the project owns the log-density.
5. Consider TensorFlow Probability's JAX substrate only where its distribution or bijector coverage materially reduces custom implementation burden.
6. Keep probabilistic engines isolated behind optional dependencies and explicit backend capability checks.
7. Add parity fixtures that compare deterministic summaries, posterior-derived summaries, and schema payloads where appropriate.
8. Document promotion criteria for moving an inference routine from experimental to supported.

## Non-Functional Requirements

1. Base installs must remain usable without Bayesian or accelerator dependencies.
2. Probabilistic outputs must remain compatible with the functional kernel and Arrow interchange strategy.
3. Tests must be deterministic enough for CI, with stochastic checks using fixed seeds or tolerance-based summaries.
4. Runtime-heavy inference checks must be scoped so normal CI remains practical.
5. XLA-backed checks must report compilation cost separately from steady-state inference cost.

## Acceptance Criteria

1. Probabilistic expansion candidates and promotion criteria are documented.
2. At least one implementation slice has schema tests, deterministic CI fixtures, and user-facing documentation.
3. Optional dependency boundaries prevent probabilistic engines from becoming required imports.
4. Kernel payloads for posterior and uncertainty artifacts are versioned and documented.
5. XLA eligibility is evaluated and documented before choosing a non-XLA probabilistic implementation.

## Out of Scope

1. Replacing deterministic fitters with probabilistic-only implementations.
2. Making PyMC, NumPyro, Stan, JAX, or any other probabilistic runtime a base dependency.
3. Promoting broad probabilistic support without schema, diagnostics, and CI coverage.
4. Treating XLA-compiled internals as the public compatibility contract.
