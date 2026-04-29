# Architecture Modernization Roadmap

## Status

Accepted direction as of 2026-04-16.

## Implementation Status

The implementation tracks for the completed roadmap stages have been completed
and archived in Conductor as of 2026-04-30. The remaining strategic follow-ons
from the Deferred Work section are now active Conductor tracks so they can be
implemented, reviewed, and archived through the same workflow as the completed
stages.

## Goal

Progress `innovate` toward a mature, durable, broadly consumable platform by sequencing changes around a small number of architectural principles:

- Array API for numerical portability
- Arrow for durable interchange
- JAX as an optional accelerator backend
- pandas plus PyArrow as the primary Python tabular surface
- selective, not foundational, use of Polars
- Python-first API stabilization followed by thin language bindings
- Rust Core Runtime as the strategic long-term execution direction

## Roadmap Stages

### Stage 1: Stabilize the Python Surface

Objective: make the Python API explicit, versionable, and safe to build on.

- Complete the canonical public API and package-topology work.
- Finish optional dependency isolation so the base install is stable without JAX/Bayesian extras.
- Publish support tiers for stable versus experimental model families and backends.

Primary tracks:

- `Canonical Public API and Package Topology`
- `Optional Backends and Dependency Stabilization`
- `Quality Gates and Release Hardening`

### Stage 2: Define the Durable Core Contract

Objective: separate durable semantics from Python object internals.

- Make the functional kernel Array API-friendly at its numerical boundary.
- Define versioned request and response schemas.
- Introduce Arrow-compatible interchange for tabular inputs, outputs, diagnostics, and provenance.

Primary tracks:

- `Functional Kernel Contract`
- `Arrow Interchange and Schema Layer`

### Stage 3: Bindings and Plugin Readiness

Objective: let other languages target one stable contract instead of reverse-engineering Python classes.

- Expose the functional kernel to R, Julia, TypeScript, Go, and Rust.
- Prepare each implemented binding for its language package manager: npm for TypeScript, crates.io for Rust, R-universe/CRAN for R, Julia General for Julia, and versioned Go modules for Go.
- Add CI jobs for every implemented binding so schema compatibility, type checks, and package tests run before release.
- Make plugin and extension boundaries explicitly versioned.
- Reuse Arrow-compatible interchange and kernel version markers across bindings.

Primary tracks:

- `Plugin API and Stability Tiers`
- `R Bindings over the Functional Kernel`
- `Julia Bindings over the Functional Kernel`
- `TypeScript Bindings over the Functional Kernel`
- `Go Bindings over the Functional Kernel`
- `Rust Bindings over the Functional Kernel`
- `Binding Publication and Multi-Language CI`

### Stage 4: Selective Performance Upgrades

Objective: adopt faster infrastructure where it produces durable wins without destabilizing the public surface.

- Keep NumPy/SciPy as the correctness and portability baseline.
- Use JAX for accelerator-backed fitters, simulation kernels, and inference where benchmarks justify it.
- Introduce Polars only in ETL-heavy or benchmark-corpus workflows where lazy execution and query optimization materially improve throughput.
- Avoid turning XLA exports, jaxlib internals, or Polars-specific query semantics into the public contract.

Primary tracks:

- `Optional Backends and Dependency Stabilization`
- `Advanced Diffusion Inference`
- `Benchmark Corpus and Model Cards`

### Stage 5: Rust Core Runtime

Objective: promote selected functional kernel operations into Rust-backed execution without changing the public API or binding contract.

- Keep Python/NumPy/SciPy as the Python reference semantics until Rust paths pass parity tests.
- Start with schema-driven operations such as `discover_models`, `predict_model`, and `simulate_model`.
- Keep fitting, diagnostics, uncertainty summaries, and optional probabilistic runtimes Python-backed until their payloads can be validated without Python object internals.
- Require schema compatibility, error mapping, and benchmark gates before any Rust-backed operation becomes the default.
- Add C# as a planned thin binding once the existing binding contract and drift checks are stable.
- Publish C# through NuGet only after the package exists, passes .NET CI, and satisfies the same schema-compatibility contract.

Primary tracks:

- `Rust Core Kernel Roadmap and C# Binding Foundation`

## Sequencing Heuristics

- Prefer contract work before wrapper proliferation.
- Prefer stable schema/versioning before cross-language SDK publication.
- Prefer Arrow-compatible boundaries before bespoke serialization.
- Prefer optional acceleration over hard backend pivots.
- Prefer Rust-backed execution behind the existing kernel contract over a second public API.
- Prefer selective DataFrame engine optimization over whole-library rewrites.

## Deferred Work

The following remain worthwhile, but they should follow the contract and interchange work rather than precede it:

- wider probabilistic inference coverage: `Probabilistic Inference Expansion`
- richer diagnostics and uncertainty tooling: `Diagnostics and Uncertainty Expansion`
- broader benchmark corpus automation: `Benchmark Corpus Automation`
- hosted services or remote execution layers: `Hosted Services and Remote Execution`
- aggressive DataFrame engine experimentation beyond ingestion and ETL edges: `DataFrame Engine Experimentation`
- broad Rust rewrites before operation-level parity and benchmark gates exist: `Rust Core Expansion`
- C# package publication before the thin-binding contract is validated: `C# Package Publication`

## Active Follow-On Tracks

These tracks convert the deferred work into Conductor-managed backlog items:

- [Probabilistic Inference Expansion](../conductor/tracks/probabilistic_inference_expansion_20260430/)
- [Diagnostics and Uncertainty Expansion](../conductor/tracks/diagnostics_uncertainty_expansion_20260430/)
- [Benchmark Corpus Automation](../conductor/tracks/benchmark_corpus_automation_20260430/)
- [Hosted Services and Remote Execution](../conductor/tracks/hosted_remote_execution_20260430/)
- [DataFrame Engine Experimentation](../conductor/tracks/dataframe_engine_experimentation_20260430/)
- [Rust Core Expansion](../conductor/tracks/rust_core_expansion_20260430/)
- [C# Package Publication](../conductor/tracks/csharp_package_publication_20260430/)
- [Roadmap Completeness Audit](../conductor/tracks/roadmap_completeness_audit_20260430/)

The `Roadmap Completeness Audit` track exists to check for implied work that is
not explicit in this roadmap, including release governance, CI/CD coverage,
observability, versioning, security, documentation, and package publication
across the supported language ecosystem.

## Decision Links

- [ADR 0001: Array API and Arrow Foundation](./adr/0001-array-api-and-arrow-foundation.md)
- [ADR 0002: JAX Is an Optional Accelerator Backend](./adr/0002-jax-is-an-optional-accelerator-backend.md)
- [ADR 0003: Python DataFrame Strategy](./adr/0003-python-dataframe-strategy.md)
- [ADR 0004: Core API, Bindings, and Rust Core Trajectory](./adr/0004-core-api-bindings-and-rust-core-trajectory.md)
