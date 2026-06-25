---
title: Probabilistic Inference
description: Posterior payload boundaries and optional probabilistic backends.
---

# Probabilistic Inference

`innovate` exposes probabilistic inference through optional XLA-aligned
engines and a stable posterior payload contract. The base package must import
without JAX, NumPyro, BlackJAX, TensorFlow Probability, or ArviZ installed.

## Backend Roles

The preferred probabilistic stack is:

* **NumPyro** for first-class probabilistic modelling on JAX.
* **BlackJAX** for lower-level samplers when `innovate` owns the log-density
  or transition logic.
* **TensorFlow Probability's JAX substrate** only when distribution or bijector
  coverage materially reduces custom code.

These engines are optional and XLA-eligible. They must not leak JAX tracing
objects, `jaxlib` internals, or XLA export artifacts into public schemas.

## Posterior Payload Contract

The `innovate.probabilistic` module defines `PosteriorSamplesPayload` for
portable posterior draws. The payload includes:

* `schema_version` for compatibility checks,
* `model_key` and `parameter_names` for binding consumers,
* `draw_shape` as `chains x draws`,
* flattened per-parameter samples,
* `engine` and `backend` provenance,
* optional seed and metadata fields.

The payload can be converted into the shared
`fitters.diagnostics_contract.UncertaintySummary` so diagnostics, bindings,
and Arrow-compatible interchange can consume posterior summaries without
depending on the sampler runtime.

## Promotion Criteria

A probabilistic implementation can move beyond experimental status only when it
satisfies these gates:

* deterministic fixtures using fixed seeds or stable summary tolerances,
* parity against deterministic summaries where that comparison is meaningful,
* versioned request, posterior, uncertainty, diagnostics, and provenance
  payloads,
* structured errors for missing optional dependencies,
* compile-time and steady-state runtime reporting for JAX/XLA paths,
* documentation that names the supported model family and backend.

The first implemented slice is the schema-compatible posterior payload plus
optional backend metadata. Full NumPyro model-family expansion remains a
follow-on implementation step.

Canonical source:

- `docs/astro-site/src/content/docs/roadmap/probabilistic-inference.md`
