---
title: Probabilistic Inference
description: Posterior payload boundaries and optional probabilistic backends.
---

# Probabilistic Inference

`innovate` exposes probabilistic inference through optional XLA-aligned engines and a stable posterior payload contract.

Preferred optional engines:

- NumPyro for first-class probabilistic modeling.
- BlackJAX for lower-level samplers under explicit log-density ownership.
- TensorFlow Probability JAX substrate when it materially simplifies distribution and bijector coverage.

## Posterior Payload Contract

The payload includes:

- `schema_version`, `model_key`, and `parameter_names`.
- `draw_shape` as `chains x draws`.
- flattened per-parameter samples plus backend metadata and optional seeds.

Promotion criteria require deterministic parity checks, schema compatibility, explicit errors for missing optional dependencies, and benchmark evidence for promoted slices.

Migration source:

- `docs/source/probabilistic_inference.rst`

