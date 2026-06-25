---
title: Remote Execution
description: Remote service boundary for kernel requests and responses.
slug: latest/operations/remote-execution
---

# Remote Execution

Remote execution is a hosted-service boundary around the functional kernel. It
does not introduce a second modeling API. Remote adapters accept versioned
kernel requests, attach service context, and return the same kernel response
schema with additional provenance and observability fields.

## Contract

Remote requests use `RemoteExecutionRequest`:

* `schema_version`: must match the kernel schema version.
* `kernel_request`: a serialized `KernelRequest`. Payloads should continue to
  use the kernel JSON and Arrow-compatible interchange contracts.
* `context`: correlation, tenancy, principal, authorization scope, trace, and
  retention metadata.
* `metadata`: optional adapter metadata. Hosted implementations must not
  require clients to depend on infrastructure-specific metadata for normal
  results.

Remote responses use `RemoteExecutionResponse`:

* `schema_version`: the kernel schema version used by the response.
* `status`: `ok` or `error`.
* `kernel_response`: the original `KernelResponse` with structured errors when
  execution is denied or fails.
* `provenance`: execution location, runtime, backend, accelerator placement,
  bridge fallback, and data retention metadata.
* `observability`: request correlation fields and timing fields for logs,
  traces, and metrics.

## Eligible Operations

The default hosted policy allows read-only or deterministic kernel operations:

* `discover_models`
* `predict_model`
* `simulate_model`
* `summarize_model`
* `diagnose_model`

`fit_model` is local-only by default because fitting can be longer running,
resource intensive, and more likely to carry sensitive training data. A hosted
deployment may allow it later, but only after it has explicit resource limits,
tenant isolation, data-retention controls, and benchmark gates.

## Threat Model And Controls

Hosted execution must authenticate callers before dispatch. Authorization is an
allow-list check over each requested operation and required scope. Every request
must include a `tenant_id` and `principal` so a hosted adapter can enforce
tenant isolation, quota, audit logging, and data-retention policy.

The first supported retention policy is `ephemeral`. Persisted inputs, outputs,
diagnostics, traces, or artifacts require an explicit service policy. Remote
adapters must block arbitrary code execution, unversioned schemas, and payloads
that bypass the kernel or Arrow-compatible interchange contracts.

## Observability

Hosted adapters must emit structured logs, traces, and metrics with these
fields:

* `request_id`
* `tenant_id`
* `principal`
* `trace_id`
* `operation`
* `duration_ms`
* `status`

The in-process adapter includes those fields in every response so tests can
validate the contract without provisioning hosted infrastructure.

## Backend Provenance

Remote execution reports backend provenance separately from model results.
Supported runtime labels are:

* `NumPy/SciPy`
* `JAX/XLA`
* `Rust-native`
* `bridge fallback`

XLA internals are not a public contract. Clients consume kernel schemas and
Arrow-compatible payloads, while provenance records whether hosted execution
used JAX/XLA, Rust-native execution, or a bridge fallback.

## Local Adapter

`InProcessRemoteExecutor` is a test adapter for the service contract. It
executes allowed requests through the local functional kernel, preserves kernel
schema versions, records structured denial errors, and attaches observability
and provenance fields.

Migration source:

* `docs/astro-site/src/content/docs/operations/remote-execution.md`
