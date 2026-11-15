---
title: Runtime and execution choices
description: Use the current Rust core, select supported native operations, and understand language, engine and parallelism boundaries.
---

Innovate's 0.7 development API uses a Rust scientific core. The earlier global
`set_backend` interface and NumPy/JAX/Numba backend instructions belong to the
[versioned archive](/innovate/latest/user-guide/backends/). They do not describe
the maintained native package.

## Separate language, algorithm and execution engine

| Choice | Current meaning | Where to look |
| --- | --- | --- |
| Python, Julia or R | Host-language access to the native implementation | [Binding overview](/innovate/bindings/) |
| Model key | Scientific model requested for a supported operation | [Model discovery](/innovate/api/generated/execute/) and [native capabilities](/innovate/native-migration/) |
| Fitting or sampling operation | A particular estimation algorithm and its payload contract | [Fitting](/innovate/user-guide/fitting/) and [Bayesian tutorial](/innovate/tutorials/bayesian-fitter/) |
| Simulation engine | A supported native trajectory mechanism for that model and task | [Scenario tutorial](/innovate/tutorials/scenario-experiment-workflows/) |
| Batch workers | Native batch concurrency for independent payloads | [fit_batch](/innovate/api/generated/fit-batch/) and [predict_batch](/innovate/api/generated/predict-batch/) |

Choosing R instead of Python does not choose a different numerical backend.
Choosing a stochastic simulation instead of a deterministic curve can change the
scientific question and output interpretation, even when the model name is
related. Inspect the documented operation and engine configuration before
comparing results.

## Table execution and parallel work

Python table operations use Polars and the [Arrow IPC boundary](/innovate/core/arrow-interchange/).
Lazy input is collected at execution. Polars expressions before that boundary
are not a promise that the scientific model runs lazily inside a query plan.

`fit_frames` and `predict_frames` process an iterable of frames sequentially and
retain documented per-input failures as values. Native `fit_batch` and
`predict_batch` instead accept explicit request payload batches and worker
control. Inspect every item in either result. The checked batch example in the
[native guide](/innovate/native-migration/) compares one and two workers and
requires stable ordering and equal numerical results for its supported fixture.

## Reproducibility and performance

Record the source or package version, model options, time grid, input checksum,
operation controls and seeds. A seed alone does not specify an analysis. For
stochastic workflows, keep the chosen engine and initial state with the result;
the scenario tutorial preserves the replay request for each execution.

The maintained documentation does not provide a validated cross-backend speedup
table or infer GPU/TPU/HPC support from historical implementations. Hardware,
dataset, compilation profile, serialization costs and scientific workload affect
timing. Report an observed benchmark with its exact configuration before making
a performance claim. Scientific equivalence and reproducibility need their own
checks when an execution path changes.

## When a capability is unavailable

An unsupported operation, model option or engine should fail visibly. Use the
[current native capability guide](/innovate/native-migration/) to determine
whether the task is implemented and qualified; an old example or optional-extra
name is not evidence that it is available in the current package.

Trusted extension callbacks can run host-language application code, as described
by the [extension API](/innovate/api/generated/invoke-registered-extension/).
Their existence does not create an automatic numerical fallback or a sandbox.
