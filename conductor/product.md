# Innovate: modelling and predicting innovation

## Purpose and origin

Innovate is a general scientific library for modelling and predicting innovation. It began with policy diffusion and extends that foundation to adoption, competition, substitution, expectations and hype, failure, abandonment, networks and innovation ecosystems. Medicines utilisation is a flagship application, not a restriction of the library's scope.

The maintainer's 7 November 2026 clarification supersedes earlier descriptions of a Python-first library, a voluntary long-term Rust migration, and a healthcare claim-exaggeration study. Hype means the influence of attention, expectations, enthusiasm and disappointment on innovation dynamics; it does not require claims to be false or excessive.

## Mandatory target architecture

Rust owns supported scientific computation. Python, R, Julia and other languages are interfaces to Rust, not alternative model implementations. Polars is the tabular computation interface; Arrow/Parquet and versioned metadata provide interchange. Pandas and xarray are not supported public inputs, runtime dependencies, transitive supported-extra dependencies or hidden conversion paths in the next breaking release. Historical releases remain historical; this planning change does not claim that existing source has already migrated.

Unsupported native operations return structured errors. No Python subprocess, Mesa, NumPy/JAX mathematical implementation or host-language simulator may silently substitute for missing Rust capability. Independent Python/R/C++ references may run in isolated test-oracle environments; they are not production dependencies. Dense native arrays are appropriate inside numerical kernels; Polars need not be forced into every scalar calculation.

## Scientific scope

- Policy, technology and product diffusion, including heterogeneous and network adoption.
- Competition, substitution, cannibalisation, complements, entry/exit and successive generations.
- Expectations and hype: composite-curve fitting, uncertainty, positive and negative uptake effects, feedback and repeated waves.
- Repeat use, persistence, switching, abandonment and re-adoption when the observation model and data support them.
- Innovation-specific strategic behaviour, market mechanisms and policy experiments, delegating general game solvers to the ecosystem's game-runtime capability.
- Estimation, prediction, simulation, identifiability diagnostics and falsifiable model comparison for these processes.

General-purpose simulation infrastructure, generic equilibrium solvers, value-of-information algorithms, terminology/data acquisition and application-specific clinical or economic assumptions belong to their authoritative providers. Generic causal machinery is not a second product: use qualified adapters/oracles and honest identification contracts where innovation questions need it.

## Ownership and integration

Innovate consumes Kairos for DES/ABM execution: one native clock, state, event ordering and simulation RNG. Innovate owns innovation behaviours and model semantics. Kairos remains independently installable and does not depend on Innovate.

Voiage can consume Innovate forecasts and joint uncertainty through an optional integration adapter and explicit economic/utility mapping. Neither core imports the other in both directions. A study or policy selected by Voiage can return as a versioned design artefact to Innovate through application orchestration; data feedback is not a dependency cycle.

Generic game representation/solution is incubated in `new-drug-reimbursement-game` and overlaps Kairos's existing game crates. Consolidate it into one independently consumable Rust game-runtime capability, keeping Kairos execution adapters optional and UOGTO semantic identifiers explicit. Innovate constructs innovation/competition games; the reimbursement application supplies its own institutional assumptions.

`vop_poc_nz` is a proof of concept. Its reusable decision, perspective, equity and VOI methods belong in Voiage after verification. `tobac`, CareOps and other applications retain their scientific scenarios and evidence while promoting reusable methods to the corresponding library. Do not copy application code into a library without tests, provenance and a consumer migration.

## Delivery status and evidence

Documentation: Astro/Starlight under `docs/astro-site` is the active documentation surface. Historical Sphinx material is archival, not an active documentation system.

The 0.7 development branch has a bounded Rust/Polars runtime with Python, Julia and R bindings only. All current and planned capabilities remain obligations in the [active migration plan](tracks/scientific_ecosystem_alignment_20261029/migration-plan.md) and [feature ledger](../migration-ledger.json). Project-owned code uses Apache-2.0.

The [programme](frontier-programme-20261107.md), [execution contract](execution-contract-20261107.md), six active tracks, [taskbook](innovation-taskbook-20261107.md) and [release gates](release-gates-20261107.md) govern delivery. Historical archived completion titles describe their recorded scope, not current scientific acceptance.
