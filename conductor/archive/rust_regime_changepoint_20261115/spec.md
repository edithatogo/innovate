# Specification: Rust Regime Detection and Changepoint Modernization (`regime.rs`)

## Objective
Modernize `rust/crates/innovate-core/src/regime.rs` by evaluating `augurs-changepoint` / `changepoint` crates (or upgrading `detect` from $O(n^2)$ exhaustive DP capped at 2,000 observations to pruned PELT with exact Welford/Chan L2 segment cost parity) and delegating per-regime curve fitting (`curve(...)`) to the `rust_linalg_optimization_20261115` stack.

## Scope
- Benchmark `augurs-changepoint` / `changepoint` against `regime::tests::native_grid_matches_independent_ruptures_fixtures` (exact changepoint index parity on L2 penalty + `jump` subsampling).
- If an external crate cannot guarantee exact `jump`/Welford L2 parity, implement exact PELT pruning in `regime::detect` while migrating `curve(...)` fitting to the shared `nalgebra`/`levenberg-marquardt` solver.
- Evaluate raising or removing the `y.len() > 2000` observation cap with bounded runtime and memory guarantees.

## Architecture Decision
- **Exact PELT Pruning in `regime::detect`:** External crates (`augurs-changepoint`, `changepoint`) do not guarantee exact parity with `ruptures`-style `jump` grid subsampling, `min_size = 2`, incremental Welford centered SSE, and ascending-predecessor tie-breaking (`regime::tests::native_grid_matches_independent_ruptures_fixtures`).
- **Implementation:** Replaced the $O(n^3)$ unpruned DP loop in `regime::detect` with exact PELT pruning (`ActiveCandidate` incremental Welford state, strict inequality pruning with IEEE-754 safety tolerance `1e-12 * (best.abs() + sse + 1.0)`), raised `MAX_DETECT_OBSERVATIONS` from `2,000` to `100,000`, and verified exact parity against an exhaustive DP oracle (`pelt_matches_exhaustive_dp_across_step_and_tied_series`) and 5,000-point series (`pelt_handles_large_series_above_legacy_2000_cap`).
- **Per-Regime Fitting:** `regime::curve` and per-segment fitting already delegate to `KernelBinding::new().predict_model` / `fit_model`, which use the shared `nalgebra`/`levenberg-marquardt` solver from `rust_linalg_optimization_20261115`.
