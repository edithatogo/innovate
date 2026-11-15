# Track 5 statistics lane ownership — 15 November 2026

## Authority and inspected base

Dylan authorised bounded implementation in this lane. Inspected main:
`42189a223fd44c823ca11aec689ea67b11c25790`. The live open-PR and branch
inventories, relevant issue searches, track files and PR discussions showed
no competing statistics owner. This records observable ownership, not an
assertion about unpublished work.

- HTTP #618 merged as `349fb0d88b765026385f5361d096070aa2c2795f`.
- Tabular #619 merged as `1cd9af803a21cdb08705cf77a45b27346afa4b69`.
- Linalg #621 remains owned at `d49eb66aa821cc3fde5ca28c7adcec45d28ed64f`.
  Its files do not include `bayesian.rs`, `bootstrap.rs` or this track.

## Allowed files

The **Innovate — Track5 statistics migration** lane owns:

- `rust/crates/innovate-core/src/bayesian.rs`
- `rust/crates/innovate-core/src/bootstrap.rs`
- New `rust/crates/innovate-core/src/bayesian/statrs_migration_tests.rs`
- New `rust/crates/innovate-core/src/bootstrap/statrs_migration_tests.rs`
- New `rust/crates/innovate-core/tests/statrs_migration.rs`
- New `rust/crates/innovate-core/tests/fixtures/statrs_20261115/**`
- `conductor/tracks/rust_stats_distributions_20261115/**`

Sole integrator `6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53` retains `causal.rs`,
all Cargo manifests/locks, CI, common registry and integration. `numerics.rs`
remains with linalg #621. No NUTS, simulation or downstream activation is
authorised. No merge, release/publication, permission change, gate weakening
or deletion is part of this lane.

## Shared dependency request

The integrator must add this exact direct dependency to
`rust/crates/innovate-core/Cargo.toml` and resolve the shared `Cargo.lock`:

```toml
statrs = { version = "=0.19.1", default-features = false, features = ["std"] }
```

Version 0.19.1 and its feature declarations were checked against the upstream
published documentation on 15 November 2026:
[crate](https://docs.rs/crate/statrs/0.19.1),
[features](https://docs.rs/crate/statrs/0.19.1/features).
The lane needs `Normal`, `Continuous`, and `ContinuousCDF`; no `rand`,
`nalgebra`, KDE, sampler or linalg feature is requested. The existing
`kairo-ecs-rng` stream remains the sampling authority. Do not remove `libm`
without a separate whole-crate consumer audit.

Shared qualification requires the actual registry checksum and complete
resolved dependency/feature graph, `cargo check --locked -p innovate-core`,
RustSec audit, and affected/full required checks on the integrated exact head.
Lane-only or disposable validation results cannot close that prerequisite.
The common registry must eventually reconcile this track's in-progress
metadata; this lane does not edit `conductor/tracks.md`.

## Numerical boundary fixed before migration

Start with independent normal quantile/CDF/log-PDF and domain/tail fixtures,
then migrate the private rank-normalisation quantile implementation. Preserve
existing R-hat, ESS, empirical percentile, missingness and deterministic RNG
conventions and their existing tolerances.

The Gaussian prior and likelihood expressions are deliberately unnormalised
log kernels. Subtracting normalised log PDFs can lose near-centre scores;
their arithmetic remains unless an independently tested replacement preserves
the finite numerical domain. `bootstrap.rs` contains empirical percentiles,
not a normal/Student-t density or CDF helper; they must retain the established
linear interpolation convention.

## Capacity and evidence limits

The main filesystem had approximately 6 MB available. A separate `/tmp`
checkout and bounded build workspace use independently available temporary
capacity; no other writer's files, caches or worktrees are removed or changed.
Capacity must be checked again before builds. Synthetic mathematical fixtures
establish only their declared numerical scope. Scientific release acceptance,
polyglot/platform qualification and planning-only downstream work remain open.
