# Plan: Rust Regime Detection and Changepoint Modernization (`regime.rs`)

- [ ] Evaluate `augurs-changepoint` and `changepoint` crates against `regime::tests::native_grid_matches_independent_ruptures_fixtures` for exact `jump`-grid and L2 Welford cost parity.
- [ ] Decide between crate adoption (if parity holds) vs exact PELT pruning in `regime::detect` and record the decision in `spec.md`.
- [ ] Migrate per-regime `curve(...)` fitting in `regime.rs` to the shared `rust_linalg_optimization_20261115` solver.
- [ ] Add large-series (`> 2000` observations) performance and numerical-stability tests and verify `cargo test --locked`, `pytest tests/native`, and `cargo doc`.
