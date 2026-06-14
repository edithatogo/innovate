# Architecture Modernization Roadmap

## Status

Accepted direction as of 2026-04-16.

## Implementation Status

The original roadmap stages are covered by Conductor records. Stage work,
deferred follow-on tracks, and the ecosystem gap tracks registered by the audit
have been completed and archived.

The product vision is not treated as fully complete while documented
future-state boundaries remain. The remediation work for those boundaries is
now captured in archived Conductor records:

- [Vision and Roadmap Truth Audit](../conductor/archive/vision_roadmap_truth_audit_20260614/)
- [Rust-Native Canonical Operation Completion](../conductor/archive/rust_native_operation_completion_20260614/)
- [Rust-Native Payload and Model-Family Coverage](../conductor/archive/rust_native_payload_model_coverage_20260614/)
- [Starlight Cutover and Legacy Cleanup](../conductor/archive/starlight_cutover_legacy_cleanup_20260614/)
- [External Submission Blocker Closure](../conductor/archive/external_submission_blocker_closure_20260614/)
- [Conductor Registry Hygiene](../conductor/archive/conductor_registry_hygiene_20260614/)

New roadmap-level gaps should become Conductor tracks before implementation
begins.

The next maturity layer is tracked in the
[Scientific and HPC readiness roadmap](source/scientific_hpc_readiness_roadmap.rst).
It registers follow-on tracks for scientific community submission readiness,
HPC packaging, accelerator evidence, Rust migration execution, ABI strategy,
polyglot documentation architecture, and governance.

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
- Prefer XLA-backed libraries for eligible accelerator work: JAX for compiled
  array kernels, NumPyro for first-class probabilistic inference, BlackJAX for
  lower-level samplers, TensorFlow Probability's JAX substrate for selected
  distribution or bijector coverage, and Diffrax for JAX-compatible differential
  equation workflows.
- Prefer explicit XLA eligibility and rejection gates before implementing
  non-XLA acceleration for probabilistic inference, diagnostics, simulation, or
  benchmark-sensitive kernels.
- Prefer Rust-backed execution behind the existing kernel contract over a second public API.
- Prefer selective DataFrame engine optimization over whole-library rewrites.
- Prefer API-preserving ABI strategy over exposing Rust structs, XLA internals,
  or scheduler-specific implementation details as public contracts.
- Prefer reproducible HPC packaging evidence before claiming HPSF, E4S, Spack,
  or EasyBuild readiness.

## Deferred Work

The following remain worthwhile, but they should follow the contract and interchange work rather than precede it:

- wider probabilistic inference coverage: `Probabilistic Inference Expansion`
- richer diagnostics and uncertainty tooling: `Diagnostics and Uncertainty Expansion`
- broader benchmark corpus automation: `Benchmark Corpus Automation`
- hosted services or remote execution layers: `Hosted Services and Remote Execution`
- aggressive DataFrame engine experimentation beyond ingestion and ETL edges: `DataFrame Engine Experimentation`
- broad Rust rewrites before operation-level parity and benchmark gates exist: `Rust Core Expansion`
- C# package publication before the thin-binding contract is validated: `C# Package Publication`

## Roadmap Coverage Map

Every roadmap goal, stage, primary track, deferred item, and ADR is mapped to a
completed Conductor archive or an explicit audit record that owns
missing-coverage checks.

### Goal Principles

| Roadmap principle | Coverage | Conductor or decision record |
| --- | --- | --- |
| Array API for numerical portability | Covered | [Functional Kernel Contract](../conductor/archive/functional_kernel_contract_20260415/), [ADR 0001](./adr/0001-array-api-and-arrow-foundation.md) |
| Arrow for durable interchange | Covered | [Arrow Interchange and Schema Layer](../conductor/archive/arrow_interchange_schema_20260416/), [ADR 0001](./adr/0001-array-api-and-arrow-foundation.md) |
| JAX as an optional accelerator backend | Covered | [Optional Backends and Dependency Stabilization](../conductor/archive/optional_backends_stabilization_20260415/), [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/), [ADR 0002](./adr/0002-jax-is-an-optional-accelerator-backend.md) |
| pandas plus PyArrow as the primary Python tabular surface | Covered | [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/), [ADR 0003](./adr/0003-python-dataframe-strategy.md) |
| selective, not foundational, use of Polars | Covered | [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/), [ADR 0003](./adr/0003-python-dataframe-strategy.md) |
| Python-first API stabilization followed by thin language bindings | Covered | [Canonical Public API and Package Topology](../conductor/archive/canonical_api_topology_20260415/), [Binding Publication and Multi-Language CI](../conductor/archive/binding_publication_ci_20260428/), [ADR 0004](./adr/0004-core-api-bindings-and-rust-core-trajectory.md) |
| Rust Core Runtime as the strategic long-term execution direction | Covered | [Rust Core Kernel Roadmap and C# Binding Foundation](../conductor/archive/rust_core_kernel_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/), [ADR 0004](./adr/0004-core-api-bindings-and-rust-core-trajectory.md) |

### Stage and Primary Track Coverage

| Roadmap item | Coverage | Conductor record |
| --- | --- | --- |
| Stage 1: Stabilize the Python Surface | Covered | Completed primary-track archives below |
| Complete the canonical public API and package-topology work | Covered | [Canonical Public API and Package Topology](../conductor/archive/canonical_api_topology_20260415/) |
| Finish optional dependency isolation so the base install is stable without JAX/Bayesian extras | Covered | [Optional Backends and Dependency Stabilization](../conductor/archive/optional_backends_stabilization_20260415/) |
| Publish support tiers for stable versus experimental model families and backends | Covered | [Quality Gates and Release Hardening](../conductor/archive/quality_gates_release_20260415/), [Plugin API and Stability Tiers](../conductor/archive/plugin_api_stability_tiers_20260415/) |
| Stage 2: Define the Durable Core Contract | Covered | Completed primary-track archives below |
| Make the functional kernel Array API-friendly at its numerical boundary | Covered | [Functional Kernel Contract](../conductor/archive/functional_kernel_contract_20260415/) |
| Define versioned request and response schemas | Covered | [Functional Kernel Contract](../conductor/archive/functional_kernel_contract_20260415/) |
| Introduce Arrow-compatible interchange for tabular inputs, outputs, diagnostics, and provenance | Covered | [Arrow Interchange and Schema Layer](../conductor/archive/arrow_interchange_schema_20260416/) |
| Stage 3: Bindings and Plugin Readiness | Covered | Completed binding, plugin, and publication archives below |
| Expose the functional kernel to R, Julia, TypeScript, Go, and Rust | Covered | [R Bindings over the Functional Kernel](../conductor/archive/r_bindings_kernel_20260415/), [Julia Bindings over the Functional Kernel](../conductor/archive/julia_bindings_kernel_20260415/), [TypeScript Bindings over the Functional Kernel](../conductor/archive/typescript_bindings_kernel_20260416/), [Go Bindings over the Functional Kernel](../conductor/archive/go_bindings_kernel_20260416/), [Rust Bindings over the Functional Kernel](../conductor/archive/rust_bindings_kernel_20260416/) |
| Prepare each implemented binding for its language package manager | Covered | [Binding Publication and Multi-Language CI](../conductor/archive/binding_publication_ci_20260428/) |
| Add CI jobs for every implemented binding | Covered | [Binding Publication and Multi-Language CI](../conductor/archive/binding_publication_ci_20260428/) |
| Make plugin and extension boundaries explicitly versioned | Covered | [Plugin API and Stability Tiers](../conductor/archive/plugin_api_stability_tiers_20260415/) |
| Reuse Arrow-compatible interchange and kernel version markers across bindings | Covered | [Arrow Interchange and Schema Layer](../conductor/archive/arrow_interchange_schema_20260416/), [Binding Publication and Multi-Language CI](../conductor/archive/binding_publication_ci_20260428/) |
| Stage 4: Selective Performance Upgrades | Covered | Completed optional-backend, inference, benchmark, and XLA archives below |
| Keep NumPy/SciPy as the correctness and portability baseline | Covered | [Optional Backends and Dependency Stabilization](../conductor/archive/optional_backends_stabilization_20260415/), [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/) |
| Use JAX for accelerator-backed fitters, simulation kernels, and inference where benchmarks justify it | Covered | [Advanced Diffusion Inference](../conductor/archive/advanced_diffusion_inference_20260415/), [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/) |
| Maintain benchmark corpus and model cards for scientific comparison | Covered | [Benchmark Corpus and Model Cards](../conductor/archive/benchmark_corpus_modelcards_20260415/) |
| Introduce Polars only in ETL-heavy or benchmark-corpus workflows | Covered | [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/) |
| Avoid turning XLA exports, jaxlib internals, or Polars-specific query semantics into the public contract | Covered | [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/), [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/) |
| Stage 5: Rust Core Runtime | Covered | Foundation and follow-on expansion slices completed; future Rust work should use new narrow tracks |
| Keep Python/NumPy/SciPy as the Python reference semantics until Rust paths pass parity tests | Covered | [Rust Core Kernel Roadmap and C# Binding Foundation](../conductor/archive/rust_core_kernel_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/) |
| Start with schema-driven operations such as `discover_models`, `predict_model`, and `simulate_model` | Covered | [Rust Core Kernel Roadmap and C# Binding Foundation](../conductor/archive/rust_core_kernel_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/) |
| Keep fitting, diagnostics, uncertainty summaries, and optional probabilistic runtimes Python-backed until their payloads can be validated | Covered | [Rust Core Summary and Diagnostics Migration](../conductor/archive/rust_core_summary_diagnostics_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/) |
| Require schema compatibility, error mapping, and benchmark gates before any Rust-backed operation becomes the default | Covered | [Rust Core Benchmarking and Profiling Tooling](../conductor/archive/rust_core_benchmarking_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/) |
| Add C# as a planned thin binding once the existing binding contract and drift checks are stable | Covered | [Rust Core Kernel Roadmap and C# Binding Foundation](../conductor/archive/rust_core_kernel_20260428/), [C# Package Publication](../conductor/archive/csharp_package_publication_20260430/) |
| Publish C# through NuGet only after package, .NET CI, and schema-compatibility gates pass | Covered | [C# Package Publication](../conductor/archive/csharp_package_publication_20260430/) |

The archived
[Roadmap Completeness Audit](../conductor/archive/roadmap_completeness_audit_20260430/)
track identified ecosystem fixture and governance gaps and converted them into
Conductor records. Future newly discovered architecture gaps should follow the
same rule: document the evidence, add a narrow track, and register it before
implementation begins.

## Follow-On Track Mapping

These tracks convert the deferred work and subsequent audit recommendations into
Conductor-managed implementation items. Completed follow-on tracks point to their
Conductor archive.

- [Probabilistic Inference Expansion](../conductor/archive/probabilistic_inference_expansion_20260430/)
- [Diagnostics and Uncertainty Expansion](../conductor/archive/diagnostics_uncertainty_expansion_20260430/)
- [Benchmark Corpus Automation](../conductor/archive/benchmark_corpus_automation_20260430/)
- [Hosted Services and Remote Execution](../conductor/archive/hosted_remote_execution_20260430/)
- [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/)
- [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/)
- [C# Package Publication](../conductor/archive/csharp_package_publication_20260430/)
- [Roadmap Completeness Audit](../conductor/archive/roadmap_completeness_audit_20260430/)
- [Lifecourse Adoption-Trajectory Fixture](../conductor/archive/lifecourse_adoption_fixture_20260504/)
- [Voiage Diffusion-Uncertainty Fixture](../conductor/archive/voiage_uncertainty_fixture_20260504/)
- [Operational Modeling Fixture Contracts](../conductor/archive/operational_modeling_fixtures_20260504/)
- [HEOML Schema Placement Decision](../conductor/archive/heoml_schema_placement_20260504/)
- [MARS Surrogate Benchmark Gate](../conductor/archive/mars_surrogate_benchmark_gate_20260504/)
- [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/)
- [HEOR Process Mining Fixture Contract and Interface Decision](../conductor/archive/heor_process_mining_fixture_contract_20260504/)
- [Binding Package Version and Language-Suffix Name Alignment](../conductor/archive/binding_package_version_name_alignment_20260504/)
- [Scientific and HPC Ecosystem Readiness Roadmap](../conductor/archive/scientific_hpc_ecosystem_readiness_20260507/)
- [Community Submission Readiness Matrix](../conductor/archive/community_submission_readiness_20260507/)
- [HPC Packaging and Registry Readiness](../conductor/archive/hpc_packaging_registry_readiness_20260507/)
- [Accelerator and Parallel Execution Evidence](../conductor/archive/accelerator_parallel_execution_evidence_20260507/)
- [Rust Core Migration Execution Plan](../conductor/archive/rust_core_migration_execution_20260507/)
- [ABI and Binary Compatibility Strategy](../conductor/archive/abi_binary_compatibility_strategy_20260507/)
- [Polyglot Repository and Documentation Architecture](../conductor/archive/polyglot_docs_repo_architecture_20260507/)
- [External Governance and Sustainability Dossier](../conductor/archive/external_governance_sustainability_20260507/)

The `Scientific and HPC Ecosystem Readiness Roadmap` track records the current
and future architecture diagrams, a submission-readiness matrix, HPC gaps,
accelerator gaps, ABI boundaries, and a subagent-ready dependency graph. It
includes Apache Arrow, PyPA, pyOpenSci, rOpenSci, JOSS, NumFOCUS, HPSF, E4S,
Spack, EasyBuild, scikit-learn-contrib, .NET Foundation, and Julia/R community
readiness targets.

The `Voiage Diffusion-Uncertainty Fixture` track now points downstream VOI
examples to
`specs/ecosystem/voiage/uncertainty/diffusion_v1/manifest.json` as a
documented-stage uncertainty source; VOI method implementation and supported
adapter compatibility remain outside this fixture.

The `HEOML Schema Placement Decision` track records the interim schema home as
`specs/ecosystem/heoml/extensions/innovate/` and defers migration until a
standalone `heoml` repository provides a semver schema bundle, fixture CI,
stable namespace, and deprecation window. The placement preserves
binding-friendly JSON, JSON Schema, and Arrow-compatible payload contracts
without private Python object framing.

The `Roadmap Completeness Audit` track exists to check for implied work that is
not explicit in this roadmap, including release governance, CI/CD coverage,
observability, versioning, security, documentation, and package publication
across the supported language ecosystem.

The `HEOR Process Mining Fixture Contract and Interface Decision` track records
the first documented-stage event-log fixture contract for process-mining
ecosystem work. It keeps PM4Py reference-only, plans a CLI surface before any
runtime adapter, and defers MCP until process-mining artifacts become
agent-queryable or workflow-orchestration heavy.

The `Probabilistic Inference Expansion` track starts with a stable posterior
payload and optional backend metadata contract in
`docs/source/probabilistic_inference.rst`. Broader NumPyro model-family
coverage remains sequenced behind deterministic fixtures, schema compatibility,
and XLA promotion gates.

The `Diagnostics and Uncertainty Expansion` track starts with a versioned
diagnostics artifact payload in
`docs/source/diagnostics_uncertainty_artifacts.rst`. The first slice covers
residual diagnostics, residual-bias calibration checks, uncertainty interval
rows, and model-comparison metrics behind the existing kernel and Arrow table
contracts. JAX/XLA acceleration remains a promotion gate for future array-heavy
diagnostics rather than a public artifact format.

The `Benchmark Corpus Automation` track adds fast benchmark metadata and
model-card freshness validation. Benchmark cases record runtime tier, CI policy,
reference backend, XLA compilation cost, XLA steady-state runtime, accelerator
target, and baseline model metadata so expensive benchmark runs stay opt-in
while promotion evidence remains structured.

The `Hosted Services and Remote Execution` track adds a remote execution contract
around the functional kernel rather than a second public API. The first slice
documents the request envelope, structured errors, observability fields, tenant
and data-retention controls, and backend provenance for NumPy/SciPy, JAX/XLA,
Rust-native, and bridge fallback execution.

The `DataFrame Engine Experimentation` track keeps pandas plus PyArrow as the
default Python tabular surface while adding an optional Polars experiment behind
the Arrow-compatible kernel table contract. Benchmark fixtures record
correctness, timing, memory, fallback behavior, and whether any gain came from
tabular execution rather than XLA-backed numerical kernels.

The `XLA Backend Strategy and JAX Kernel Promotion Gates` track exists to make
the optional accelerator preference operational. It should define when JAX/XLA
is preferred, when NumPy/SciPy remains the reference path, when Rust-native
execution should compete with XLA-backed kernels, and when dynamic operational
simulation semantics should stay outside XLA-backed implementation.
The durable policy is documented in the Sphinx page
`XLA Backend Strategy` (`docs/source/xla_backend_strategy.rst`).

The `Rust Core Expansion` track now starts from an operation support inventory
rather than a broad rewrite. The inventory records native Rust scope, bridge
fallback scope, Python-only reference scope, Rust vs JAX/XLA eligibility, stable
error behavior, and the benchmark promotion dossier required before any
Rust-native path becomes the default.

## Decision Links

| Decision | Coverage | Conductor record |
| --- | --- | --- |
| [ADR 0001: Array API and Arrow Foundation](./adr/0001-array-api-and-arrow-foundation.md) | Covered | [Functional Kernel Contract](../conductor/archive/functional_kernel_contract_20260415/), [Arrow Interchange and Schema Layer](../conductor/archive/arrow_interchange_schema_20260416/) |
| [ADR 0002: JAX Is an Optional Accelerator Backend](./adr/0002-jax-is-an-optional-accelerator-backend.md) | Covered | [Optional Backends and Dependency Stabilization](../conductor/archive/optional_backends_stabilization_20260415/), [XLA Backend Strategy and JAX Kernel Promotion Gates](../conductor/archive/xla_backend_strategy_20260430/) |
| [ADR 0003: Python DataFrame Strategy](./adr/0003-python-dataframe-strategy.md) | Covered | [DataFrame Engine Experimentation](../conductor/archive/dataframe_engine_experimentation_20260430/) |
| [ADR 0004: Core API, Bindings, and Rust Core Trajectory](./adr/0004-core-api-bindings-and-rust-core-trajectory.md) | Covered | [Binding Publication and Multi-Language CI](../conductor/archive/binding_publication_ci_20260428/), [Rust Core Kernel Roadmap and C# Binding Foundation](../conductor/archive/rust_core_kernel_20260428/), [Rust Core Expansion](../conductor/archive/rust_core_expansion_20260430/), [C# Package Publication](../conductor/archive/csharp_package_publication_20260430/) |
| [ADR 0005: HEOML Schema Placement](./adr/0005-heoml-schema-placement.md) | Covered | [HEOML Schema Placement Decision](../conductor/archive/heoml_schema_placement_20260504/) |
