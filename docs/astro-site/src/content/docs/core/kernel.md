---
title: Native kernel contract
description: Understand schema-2.0 requests, native response envelopes, failure handling and reusable fitted state.
---

The native kernel is the scientific execution boundary for the 0.7 development
API. Python, Julia and R supply requests to the Rust implementation; supported
operations and model keys are defined by that implementation. A language binding
does not silently substitute an earlier numerical backend when a feature is
unavailable.

## Select an interface

| Interface | Appropriate use | Reference |
| --- | --- | --- |
| Python `fit` and `predict` | Ordered Polars tables and fitted model state | [Fit](/innovate/api/generated/fit/) and [predict](/innovate/api/generated/predict/) |
| Python `execute` | Operation-specific JSON-compatible payloads | [Execute](/innovate/api/generated/execute/) |
| Julia `kernel_call` | Explicit requests through the native C ABI | [Julia reference](/innovate/api/generated/julia/) |
| R `kernel_call` or `kernel_call_json` | R objects or a JSON response through the C ABI | [R reference](/innovate/api/generated/r/) |
| Rust workspace crates | Native scientific, tabular and FFI integration | [Rust core](/innovate/bindings/rust/) |

Use a documented high-level wrapper when it expresses your task. Low-level
requests expose operation-specific fields and therefore require closer attention
to that operation's schema. Model discovery is available through
`execute("discover_models")`; the [executable API example](/innovate/api/generated/execute/)
checks that discovery returns the expected native model inventory.

## Request and response fields

The Python `execute` wrapper builds a request with `schema_version` equal to
`"2.0"`, an `operation`, a `model_key` and a `payload`. Its optional `model`
argument becomes `model_key`; it is `null` for operations that do not select a
model. Payload fields depend on the chosen operation. JSON serialization rejects
non-finite values rather than converting them into an undocumented sentinel.

A successful call returns the decoded response envelope, including its `result`
and operation metadata. Success does not require an explicit `error` member:
some successful envelopes omit it. A non-null native error is raised by the
Python wrapper as [NativeError](/innovate/api/generated/native-error/), with a
machine-readable `code` and optional `validation` details. Serialization, Polars
and invalid table errors can arise before a native envelope exists; use the
specific wrapper's documented exception contract.

Scenario and batch operations can contain per-item outcomes. Inspect the
documented result structure instead of treating outer-request success as proof
that every scenario succeeded. The [scenario tutorial](/innovate/tutorials/scenario-experiment-workflows/)
checks every execution and comparison before exporting or replaying results.

## Preserve state and configuration

[FitResult](/innovate/api/generated/fit-result/) retains the successful fitting
envelope, including constructor options needed for subsequent predictions.
Its `predict` method uses that fitted configuration with a new explicit time
table. Preserve the envelope and relevant runtime versions when recording an
analysis; a parameter vector without its model options can describe a different
scientific calculation.

The Python result object prevents attribute reassignment, but the nested envelope
is mutable and accessors return shallow copies. Treat it as an analysis record;
do not mutate nested state to create an undocumented alternative model. Use
explicit scenario specifications when comparing alternatives.

## Validation and trust boundaries

Input validation establishes conformity with implemented rules. It does not
establish causal identification, measurement quality, a data licence, model
adequacy or empirical interval coverage. Read [core concepts](/innovate/core/)
for the distinction between a valid request and a suitable analysis.

Registered extension callbacks run trusted code in their host process.
Manifest resolution and result validation do not sandbox a callback or establish
its scientific validity. See the [extension API](/innovate/api/generated/invoke-registered-extension/)
before integrating application code.

The binding source is the implementation reference for
[request serialization and error decoding](https://github.com/edithatogo/innovate/blob/main/bindings/python/innovate/__init__.py).
Generated API manifests identify their exact source hashes. Cross-version replay
and release readiness remain explicit qualification work in the
[native guide](/innovate/native-migration/).
