# Specification: XLA Backend Strategy and JAX Kernel Promotion Gates

## Overview

Define the project-wide strategy for preferring XLA-based libraries where they fit the kernel contract. JAX should be the preferred accelerator path for suitable numerical kernels, probabilistic inference, diagnostics, simulation, and benchmarked model workflows, while NumPy/SciPy remains the reference correctness path and base installs remain free of mandatory accelerator dependencies.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- User preference: use XLA-based libraries for roadmap follow-on tracks where practical
- Existing direction: JAX is an optional accelerator backend, not a public ABI

## Functional Requirements

1. Define when a kernel should be considered XLA-eligible, including pure array semantics, static-shape expectations, deterministic PRNG handling, and limited Python object dependence.
2. Make JAX the preferred accelerator backend for eligible kernels and document NumPy/SciPy reference parity requirements.
3. Identify preferred XLA-aligned libraries by domain:
    - [ ] JAX for compiled array kernels, autodiff, vectorization, and accelerator execution.
    - [ ] NumPyro for first-class probabilistic modelling and inference over JAX.
    - [ ] BlackJAX for lower-level samplers when the project owns the log-density or transition logic.
    - [ ] TensorFlow Probability's JAX substrate only when distribution or bijector coverage materially reduces custom code.
    - [ ] Diffrax for JAX-compatible differential equation workflows.
4. Define promotion gates for XLA-backed implementations, including correctness parity, compile-time reporting, run-time benchmarks, memory behavior, fallback behavior, and schema compatibility.
5. Define rejection criteria for workflows that are too dynamic for XLA, including event-driven operational simulation paths that cannot be expressed with JAX control-flow primitives without distorting semantics.
6. Update relevant backlog tracks so probabilistic inference, diagnostics, benchmark automation, DataFrame experiments, and Rust-core migration all evaluate XLA-backed options before implementing non-XLA acceleration.

## Non-Functional Requirements

1. JAX/XLA must remain optional and isolated behind dependency extras or backend gates.
2. XLA-backed paths must not leak `jaxlib`, XLA exports, traced internals, or backend-specific query semantics into the public API.
3. Promotion decisions must include benchmark evidence that separates first-call compilation costs from steady-state execution.
4. Tests must be deterministic across CPU-only CI and tolerant of accelerator-specific numerical differences where appropriate.
5. Fallback paths must produce structured errors or NumPy/SciPy reference behavior when JAX is unavailable.

## Acceptance Criteria

1. A documented XLA/JAX eligibility and promotion policy exists.
2. Relevant active Conductor tracks reference the policy and include XLA evaluation gates.
3. Tests guard that roadmap and track documentation keep the XLA strategy visible.
4. The strategy preserves the optional-backend architecture and does not require JAX for base installs.

## Out of Scope

1. Making JAX, NumPyro, BlackJAX, TensorFlow Probability, or Diffrax mandatory base dependencies.
2. Replacing all NumPy/SciPy reference kernels with JAX implementations in this track.
3. Publishing XLA exports as a public compatibility contract.
4. Forcing classic discrete-event simulation semantics into XLA when the event model is inherently dynamic.
