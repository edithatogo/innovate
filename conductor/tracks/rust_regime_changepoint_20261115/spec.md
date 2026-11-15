# Specification: Rust Regime Detection and Changepoint Modernization (`regime.rs`)

## Objective
Modernize `rust/crates/innovate-core/src/regime.rs` by evaluating `augurs-changepoint` / `changepoint` crates (or upgrading `detect` from $O(n^2)$ exhaustive DP capped at 2,000 observations to pruned PELT with exact Welford/Chan L2 segment cost parity) and delegating per-regime curve fitting (`curve(...)`) to the `rust_linalg_optimization_20261115` stack.

## Scope
- Benchmark `augurs-changepoint` / `changepoint` against `regime::tests::native_grid_matches_independent_ruptures_fixtures` (exact changepoint index parity on L2 penalty + `jump` subsampling).
- If an external crate cannot guarantee exact `jump`/Welford L2 parity, implement exact PELT pruning in `regime::detect` while migrating `curve(...)` fitting to the shared `nalgebra`/`levenberg-marquardt` solver.
- Evaluate raising or removing the `y.len() > 2000` observation cap with bounded runtime and memory guarantees.
