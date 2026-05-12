---
title: DataFrame Engine Experiments
description: Stable kernel-schema-first DataFrame engine compatibility notes.
---

# DataFrame Engine Experiments

Pandas plus PyArrow remains the default tabular surface. Experimental engines must not alter kernel semantics.

Current experiment scope:

- `pandas`: production-facing ergonomics and plotting/data prep.
- `PyArrow`: default durable interchange for table payloads.
- `polars`: optional and experimental until reproducible evidence and compatibility gates justify broader support.

## Contract Boundaries

The public surface remains the functional-kernel schema and Arrow-compatible payloads. Engine-specific expression trees and scheduler-native internals are not public contract.

Benchmark evidence must report fixture shapes, runtime, memory, and whether observed gains came from table execution or kernel-level acceleration.

Migration source:

- `docs/source/dataframe_engine_experiments.rst`

