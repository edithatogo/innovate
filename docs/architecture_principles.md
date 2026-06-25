# Architecture Principles

## Status

Accepted on 2026-04-16.

## Normative Language

The key words `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` in this document are to be interpreted as described in RFC 2119 and RFC 8174.

## Purpose

These principles define the long-lived architectural direction for `innovate` as it matures from a Python library into a portable modeling platform with a stable kernel, clear compatibility boundaries, and future multi-language bindings.

## Principles

### 1. Portable Numerical Core

The public numerical contract `MUST` target Python Array API-compatible semantics rather than backend-specific behavior. Core kernels `MUST NOT` depend on NumPy-only conveniences, JAX tracing internals, or device-specific lowering details as part of the public contract.

### 2. Stable Public Boundary

The stable boundary for users and downstream bindings `MUST` be versioned, documented, and testable. Public behavior `MUST NOT` rely on Python class identity, inheritance structure, or hidden object state.

### 3. Arrow-Native Interchange

Tabular and structured interchange `MUST` use Arrow-compatible schemas and encodings where columnar data crosses package, process, or language boundaries. This is the durability layer for future R, Julia, TypeScript, Go, and Rust bindings.

### 4. Python-First, Language-Neutral Core

Python `SHOULD` remain the primary authoring and research environment. At the same time, stable execution semantics `MUST` be representable without Python-specific wrappers so the functional kernel can be reused across bindings.

### 5. Optional Acceleration

JAX `SHOULD` be supported as an optional accelerator backend for performance-sensitive fitting, simulation, and inference. XLA/JAX lowering artifacts `MUST NOT` become the library's public ABI or long-term interchange format.

### 6. Deliberate DataFrame Evolution

Pandas `SHOULD` remain the primary user-facing DataFrame API for the Python surface. PyArrow `MUST` be treated as foundational infrastructure for columnar types and interchange. Polars `MAY` be introduced selectively for ETL-heavy or benchmark-ingestion workflows, but `MUST NOT` become a forced replacement for the user-facing API without a separate decision record.

### 7. Stability Tiers

Backends, model families, and kernel operations `SHOULD` declare clear stability tiers such as `stable`, `provisional`, and `internal`.
Stable surfaces `MUST` be versioned and documented, provisional surfaces `SHOULD` be treated as evolving public contract points, and internal surfaces `MUST` remain isolated from the durable public contract.

### 8. Measured Adoption of New Infrastructure

Infrastructure changes `SHOULD` be adopted only when they improve portability, maintainability, or correctness in addition to raw speed. Rewrites motivated solely by novelty or benchmark fashion `SHOULD NOT` displace working interfaces without clear lifecycle and migration benefits.

### 9. Rust Core Trajectory

Rust `SHOULD` be treated as the strategic long-term core runtime for performance-critical and portability-critical kernel execution. Rust-backed components `MUST` remain behind the stable functional kernel contract, `MUST` preserve Python reference semantics until parity tests promote them, and `MUST NOT` create a second public API that bypasses schema compatibility.

## Immediate Implications

- The durable technical foundation is `Array API + Arrow`, not `XLA` as a public surface.
- The reference numerical backend remains NumPy/SciPy.
- JAX remains an accelerator and advanced-inference option, not the universal execution contract.
- XLA-backed kernels should be preferred for eligible accelerator work when they satisfy documented promotion gates.
- The tabular strategy is `pandas + PyArrow` first, with selective Polars adoption at ingestion and analytics edges only.
- Rust is the long-term core runtime direction, but Python remains the ergonomic reference interface.

## Related Documents

- [Architecture Modernization Roadmap](./architecture_modernization_roadmap.md)
- [XLA Backend Strategy](./astro-site/src/content/docs/operations/xla-backend.md)
- [ADR Index](./adr/index.md)
- [ADR 0004: Core API, Bindings, and Rust Core Trajectory](./adr/0004-core-api-bindings-and-rust-core-trajectory.md)
