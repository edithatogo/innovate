---
title: Accelerator and Parallel Execution Evidence
description: Evidence requirements for scalable execution claims.
slug: latest/operations/accelerator-parallel-execution
---

# Accelerator and Parallel Execution Evidence

Accelerator and parallel execution claims are evidence items, not separate public APIs.

Evidence must identify:

* execution mode (CPU parallelism, GPU, TPU, distributed, scheduler-aware),
* reproducible benchmark artifact,
* baseline and hardware context,
* fallback and rejection rationale.

Public evidence remains kernel capability metadata and benchmark artifacts; internals stay in implementation-only traces.

Migration source:

* `docs/source/accelerator_parallel_execution_evidence.rst`
