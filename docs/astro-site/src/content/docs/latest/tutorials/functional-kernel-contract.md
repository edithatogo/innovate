---
title: Functional Kernel Contract
description: Language-neutral execution surface and portable kernel operations.
---

The functional kernel is the language-neutral execution surface for `innovate`.
It keeps the public contract focused on serializable envelopes and Arrow-friendly
payloads so downstream bindings can call into the library without depending on
Python object identity or backend-specific internals.

## Contract overview

The kernel publishes a versioned schema through `innovate.kernel.KERNEL_SCHEMA_VERSION`
and a fixed operation list through `innovate.kernel.KERNEL_OPERATIONS`. The current
operations are:

- `discover_models`
- `fit_model`
- `predict_model`
- `simulate_model`
- `summarize_model`
- `diagnose_model`

Request and response envelopes are represented by:

- `innovate.kernel.KernelRequest`
- `innovate.kernel.KernelResponse`
- `innovate.kernel.KernelError`

## Portable payloads

Where tabular or array-shaped data is exchanged, the kernel uses explicit payload
objects instead of exposing NumPy arrays directly:

- `innovate.kernel.KernelArrayPayload` for numeric array data
- `innovate.kernel.KernelTablePayload` for row/column tabular data
- `innovate.kernel.KernelDiscoveryResponse` for machine-readable model discovery

These payloads serialize to JSON-friendly dictionaries and are intended to map cleanly
to Arrow tables or language-native equivalents in future bindings.

## Versioning and compatibility

The kernel request and response envelopes carry a `schema_version` field. The
current contract version is `1.0`. The compatibility rule is intentionally narrow:
same-major schema versions are acceptable only when the current kernel can still
understand the envelope shape, and a major version bump requires a documented
migration path.

Binding authors should treat `schema_version` as part of the public ABI for the
functional surface. Preserve the declared version when serializing envelopes, reject
unknown major versions early, and document any additive field changes before they are
rolled out to downstream consumers.

## Relationship to the OO API

The kernel complements the Python object-oriented API rather than replacing it.
The object model remains the ergonomic layer for interactive Python workflows,
while the kernel defines the stable cross-language contract that bindings can target
directly. Discovery is already wired to the canonical capability registry; execution
adapters are added in later tracks.

## Boundary guidance

Use the functional kernel when you need a portable execution boundary or when you
are writing a non-Python binding. Use the object-oriented Python API when you want
ergonomic model objects, direct method calls, and convenience helpers in a notebook
or application script.

The kernel is the interoperability layer. It is not a separate modeling stack, and
it should not fork model behavior from the canonical Python API unless a binding or
transport constraint requires it.
