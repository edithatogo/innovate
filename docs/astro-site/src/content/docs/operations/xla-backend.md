---
title: XLA Backend Strategy
description: Eligibility and promotion policy for optional XLA execution.
---

# XLA Backend Strategy

`innovate` prefers JAX/XLA for eligible accelerator work while keeping NumPy/SciPy as the reference correctness path.

Eligibility checks:

- Schema- and array-oriented payloads.
- Bounded or static shapes.
- Deterministic operation semantics with documented behavior.
- Public outputs limited to kernel/payload layers, not JAX internals.

Preferred libraries include JAX, NumPyro, BlackJAX, and optional Diffrax when aligned with supported workflows.

Promotion gates require parity evidence, schema compatibility, and structured capability metadata before a slice is marked non-experimental.

Migration source:

- `docs/source/xla_backend_strategy.rst`

