---
title: XLA Backend Strategy
description: Eligibility and promotion policy for optional XLA execution.
slug: latest/operations/xla-backend
---

# XLA Backend Strategy

This page defines when `innovate` should prefer XLA-backed execution and how
that preference is promoted without turning JAX internals into the public
contract.

The short rule is: prefer JAX/XLA for eligible accelerator work, keep
NumPy/SciPy as the reference correctness path, and require evidence before
promoting an XLA-backed implementation.

## Eligibility

A kernel is XLA-eligible when it has most of these traits:

* Inputs and outputs can be expressed as arrays or schema payloads rather than
  mutable Python objects.
* Shapes are static or can be bounded and documented before compilation.
* Control flow can be expressed with JAX-compatible primitives such as
  `jax.lax.scan` or `jax.lax.while_loop` without changing model semantics.
* Randomness can be driven by explicit JAX PRNG keys and deterministic fixture
  seeds.
* Public outputs remain kernel schemas, Arrow-compatible payloads, or ordinary
  Python values rather than traced JAX internals.

An implementation should reject XLA for a given slice when event queues,
runtime-dependent object mutation, unbounded shape changes, or Python callback
semantics would distort the model. Classic discrete-event simulation can still
belong in the ecosystem, but it must document whether a JAX vectorized
replication or batched Monte Carlo formulation is semantically equivalent.

## Preferred Libraries

Use these libraries by default when the eligibility checks pass:

* **JAX** for compiled array kernels, autodiff, vectorization, and CPU/GPU/TPU
  execution.
* **NumPyro** for first-class probabilistic modelling and inference on JAX.
* **BlackJAX** for lower-level samplers when `innovate` owns the log-density,
  transition, or inference loop.
* **TensorFlow Probability's JAX substrate** when distribution or bijector
  coverage materially reduces custom implementation code.
* **Diffrax** for JAX-compatible differential equation workflows.

NumPy/SciPy remains the reference path for correctness, portability, and base
install behavior. XLA-backed paths must be optional and isolated behind extras
or backend capability gates.

## Promotion Gates

An XLA-backed implementation can move from experimental to supported only when
it satisfies these gates:

* **Reference parity:** the implementation matches NumPy/SciPy reference
  semantics within documented tolerances.
* **Schema compatibility:** request, response, diagnostic, provenance, and error
  payloads remain compatible with the functional kernel contract.
* **Benchmark evidence:** reports separate first-call compilation cost from
  steady-state runtime and include the accelerator target.
* **Fallback behavior:** missing JAX or accelerator dependencies produce
  structured errors or fall back to the reference path where that is supported.
* **Deterministic tests:** stochastic behavior uses explicit PRNG keys, fixed
  seeds, or tolerance-bounded summary checks.
* **No ABI leakage:** XLA lowering details, `jaxlib` internals, and exported JAX
  artifacts do not become the public compatibility contract.

## Benchmark Reporting

Benchmark artifacts for XLA-backed paths should record:

* backend name and version,
* accelerator target,
* first-call compilation time,
* steady-state runtime,
* memory behavior where measurable,
* NumPy/SciPy reference runtime,
* Rust-native runtime when a Rust implementation is also a candidate.

This lets Rust-core work, probabilistic inference, diagnostics, and simulation
tracks compare real deployment tradeoffs instead of comparing a compiled path
against an uncompiled reference unfairly.

## Track Integration

Future Conductor tracks should apply this policy as follows:

* Probabilistic inference should prefer NumPyro or BlackJAX for eligible kernels
  before selecting a non-XLA probabilistic runtime.
* Diagnostics and uncertainty work should evaluate JAX-compatible artifact
  generation for array-heavy summaries.
* Benchmark corpus automation should validate compile-time and steady-state
  runtime fields for JAX/XLA reports.
* DataFrame experiments should distinguish tabular-engine performance from
  XLA-backed numerical-kernel performance.
* Rust-core promotion should compare native Rust, NumPy/SciPy reference, and
  eligible JAX/XLA-backed implementations before selecting a default path.
* Hosted or remote execution should record backend provenance without requiring
  clients to understand XLA internals.

The strategy follows ADR 0002: JAX is an optional accelerator backend, not the
public ABI.

Migration source:

* `docs/astro-site/src/content/docs/operations/xla-backend.md`
