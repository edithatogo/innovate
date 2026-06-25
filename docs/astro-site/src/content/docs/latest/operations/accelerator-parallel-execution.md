---
title: Accelerator and Parallel Execution Evidence
description: Evidence requirements for scalable execution claims.
slug: latest/operations/accelerator-parallel-execution
---

# Accelerator and Parallel Execution Evidence

Innovate treats accelerator and parallel execution as implementation evidence,
not as a new public API surface. Public contracts continue to be the functional
kernel schema, Arrow-compatible interchange, documented diagnostics outputs, and
capability metadata. Backend internals such as XLA lowering forms, jaxlib
objects, vendor runtime handles, Rust native structs, and scheduler-internal job
identifiers must not appear in public APIs, persisted evidence artifacts, or ABI
claims.

## Evidence Scope

Every performance or scalability claim must identify one execution mode and link
to a reproducible artifact that records the benchmark command, environment,
comparison baseline, memory observation, fallback status, and rejection
rationale where applicable. Claims about distributed execution and
scheduler-aware benchmarking follow the same evidence gate as local accelerator
claims.

| Execution mode | Evidence required before promotion | Public contract boundary |
| --- | --- | --- |
| CPU parallelism | Vectorized NumPy/SciPy and native CPU paths record CPU model, core count, thread settings, compile or warm-up costs when present, steady-state runtime, and memory observations. | Expose only kernel capability metadata and benchmark artifacts, not thread-pool internals or native struct layouts. |
| GPU | JAX/XLA-backed GPU claims record accelerator target, backend version, first-call compile time, steady-state runtime, memory pressure, fallback behavior, and a CPU comparison for the same kernel payload. | Expose optional backend capability and evidence links, not XLA HLO, device buffers, or jaxlib internals. |
| TPU | TPU evidence is limited to kernels with stable array shapes and clear transfer costs. Artifacts record TPU type, runtime family, compile time, steady-state runtime, memory behavior, and rejection rationale for dynamic or unsupported workloads. | Expose eligibility and deferral status only; TPU runtime details remain implementation metadata. |
| ASIC-oriented and vendor-specific accelerator | ASIC-oriented or other vendor-specific accelerator evaluation records the vendor runtime family, target class, benchmark command, comparison baseline, and fallback or rejection status before any claim is made. | Expose backend-neutral capability metadata only. Vendor ABI details, handles, generated code, and hardware-specific calling conventions remain private implementation evidence. |
| Distributed execution | Distributed execution evidence records scheduler family, node and worker shape, task counts, data movement notes, wall-clock runtime, memory observations, and single-node comparison. | Expose reproducible evidence and capability metadata, not cluster-private identifiers or scheduler-specific control-plane objects. |
| Scheduler-aware benchmarking | Scheduler-aware benchmarking records queue or partition class, allocation shape, walltime request, runner command, Slurm or PBS submission wrapper when used, and whether results were accepted, deferred, or rejected. | Expose scheduler family and anonymized allocation shape only. Internal Slurm, PBS, or site-specific job identifiers are not public fields. |

## Artifact Requirements

Machine-readable evidence must follow
`docs/source/_static/accelerator_parallel_execution_evidence_schema.json`.
At minimum, each artifact records:

* `execution_mode` and `accelerator_target`.
* `scheduler` with `none` for local CPU, GPU, TPU, or vendor runs that do not
  use a batch scheduler.
* `runner_command` and enough environment metadata to reproduce the run.
* `compile_time_seconds` when compilation, lowering, graph capture, or warm-up
  is part of the backend.
* `steady_state_runtime_seconds` for the measured workload after setup costs.
* `memory_observation` for host, device, and distributed memory pressure.
* `fallback_status` with one of `accepted`, `fallback_used`, `deferred`, or
  `rejected`.
* `rejection_rationale` for unsupported accelerators, dynamic shapes, scheduler
  constraints, missing optional dependencies, or unavailable hardware.
* `evidence_uri` pointing to the benchmark output, CI artifact, or manually
  reviewed dossier entry.

## Runner Expectations

Benchmark runners must separate setup from steady-state measurements where the
backend has compilation or placement costs. GPU and TPU runners should record
the selected backend through backend-neutral labels such as `gpu` or `tpu` and
may record backend versions as artifact metadata. They must not serialize XLA
lowering text, jaxlib objects, device pointers, or generated vendor code into
public artifacts.

Distributed and scheduler-aware runners should keep scheduler data portable:
record scheduler family, anonymized queue or partition class, node count, task
count, worker count, requested walltime, and runner command. The distributed
execution evidence must preserve the same backend-neutral kernel contract used
by local runs. Slurm and PBS examples are acceptable evidence wrappers when the
artifact omits site-private job identifiers and cluster-specific account names.

## Promotion Policy

An execution mode can be referenced in publication, packaging, or community
submission text only when the evidence artifact exists and the fallback status
is `accepted` or clearly explains the narrower claim. Unsupported accelerators
remain documented as `deferred` or `rejected` until an artifact demonstrates that
the same backend-neutral kernel contract works on that target.

Migration source:

* `docs/astro-site/src/content/docs/operations/accelerator-parallel-execution.md`
