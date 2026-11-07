# Tech Stack

## Mandatory target: Rust scientific core and Polars

The maintainer's 7 November 2026 direction replaces the Python-first migration trajectory. Supported scientific model evaluation, inference and simulation execute Rust. Python/R/Julia/other interfaces marshal native calls, configure jobs and present results. Missing native capability returns an explicit error.

Polars is the supported tabular computation surface; Arrow/Parquet plus typed metadata is the interchange boundary. Pandas/xarray APIs, production imports, hidden conversions and transitive supported-extra dependencies are retired in the next breaking release. Separate external test-oracle environments may use established scientific packages, but are unreachable from production dispatch. The current source remains hybrid until the migration tasks pass; this policy is not a claim that the code has already changed.

Target paths: `crates/innovate-core`, `crates/innovate-data`, `crates/innovate-arrow`, optional `crates/innovate-kairos` and `crates/innovate-games`, direct Python/PyO3 and versioned native FFI bindings. Minimal numerical kernels may use native dense arrays; they need not materialise a dataframe.

## Dependencies and tooling

Use [dated dependency candidates](dependency-candidates-20261107.md). Rust 1.99.1 and Python 3.15.2 interface/tooling are starting qualification targets, not fabricated completed upgrades. Match Kairos's exact accepted Polars/Arrow dependency closure or qualify a newer replacement with its consumers. Retain required root patches and package-publication constraints. Arrow Rust, Polars Rust/Python and PyArrow are separate release lines.

Separate core, bindings, optional backend, build, isolated test-oracle and development dependencies. Qualify maintained numerical tools before replacing custom code. No production JAX/NumPy/BlackJAX/Mesa execution fallback; GPU acceleration must still satisfy native execution and actual hardware evidence.

Python dependency management remains **uv**-first, with **nox** — Python task orchestration for repeatable local and CI sessions. Rust uses Cargo and an exact toolchain/lock; no undeclared latest or rolling Git main.

## Simulation and ecosystem

Kairos owns simulation time, events, state, resources and simulation RNG. Innovate owns innovation model semantics and behaviours. Optional game adapters consume the single generic game-runtime capability, not duplicate solvers. Voiage consumes exported joint uncertainty through explicit decision bindings; application artefacts may feed selected designs back without mutual core dependencies.

## Documentation

- **Site Generator**: Astro with Starlight; `docs/astro-site` is the active site.
- **Versioned Baseline**: existing `astro` `^7.0.5`, `@astrojs/starlight` `^0.41.3`, and `@astrojs/markdown-remark` `^7.2.0` records are historical/current-manifest observations, not an instruction to freeze obsolete versions.
- **Polyglot Plugin**: starlight-polyglot; versioned pages use starlight-versions `0.9.1`; link tooling includes starlight-links-validator `0.25.2`.
- **Search**: `@astrojs/starlight-docsearch` `0.7.0` remains a separately qualified integration.
- **Prose Linting**: Vale with Repo/ValueProse; dependency updates require their own build evidence.
- **Build Gate**: run `pnpm build && pnpm check` in the actual docs workspace; do not claim it ran from this plan.
- Static historical release evidence under `docs/source/_static/` remains source data, not current acceptance by itself.
- **TypeScript Binding Toolchain**: existing Node.js 26, TypeScript `^6.0.3`, Vitest `^4.1.9` and `@types/node` `^26.0.0` declarations must be reconciled with qualified current candidates before release.

## Historical Rust ownership records

`Rust Core Full Native Migration and Ownership Closure` and `Rust Core Migration Completion and Polyglot Claim Closure` are historical titles. They do not establish present full native ownership. The current source remains hybrid and the [Rust-core convergence track](tracks/rust_core_convergence_20261107/index.md) owns the remaining acceptance.

For historical profiling context, **cargo-flamegraph** and **DHAT** are recorded Rust profiling tools. Historical **XLA-Backed Libraries** evidence is retained as external-reference/experimental history, not permission for a supported non-Rust production engine. All performance claims require current measured evidence.

## Delivery

Follow the [programme](frontier-programme-20261107.md), [taskbook](innovation-taskbook-20261107.md), [execution contract](execution-contract-20261107.md) and [release gates](release-gates-20261107.md). No runtime dependency pins are modified by this planning revision.
