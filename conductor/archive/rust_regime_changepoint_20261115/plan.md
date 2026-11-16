# Plan: Rust Regime Detection and Changepoint Modernization (`regime.rs`)

> **Execution Order:** #9 (Run after `rust_linalg_optimization_20261115` so per-regime `curve(...)` fitting uses the shared linalg/optimization solver).

## Phase 1: Crate Evaluation & Decision Gate
- [x] **1.1 Evaluate `augurs-changepoint` and `changepoint`:** Evaluated against `regime::tests::native_grid_matches_independent_ruptures_fixtures`; external crates lack exact `jump`-grid, `min_size=2`, Welford SSE, and ascending-predecessor tie parity.
- [x] **1.2 Record architecture decision in `spec.md`:** Documented decision to implement exact PELT pruning directly in `regime::detect`.
- [x] **1.3 Add large-series (`> 2000` observations) benchmark/stability tests:** Added `pelt_matches_exhaustive_dp_across_step_and_tied_series` and `pelt_handles_large_series_above_legacy_2000_cap` (5,000 observations) in `rust/crates/innovate-core/src/regime.rs`.
- [x] **1.4 Conductor Review (Phase 1):** Verified decision and characterization tests.

## Phase 2: Changepoint Detection & Per-Regime Solver Migration
- [x] **2.1 Implement chosen changepoint detector in `regime.rs`:** Implemented exact PELT pruning with incremental Welford state in `regime::detect`, raised cap to `MAX_DETECT_OBSERVATIONS = 100_000`, and verified `cargo test -p innovate-core --locked regime` (3 unit tests, 3 `tests/regime.rs` tests, 1 `tests/native_boundary.rs` test passed).
- [x] **2.2 Migrate per-regime `curve(...)` fitting:** Confirmed `regime::curve` and segment fitting already delegate through `KernelBinding` to the shared `nalgebra`/`levenberg-marquardt` solver (`rust_linalg_optimization_20261115`).
- [x] **2.3 Conductor Review (Phase 2):** Verified `regime` unit and integration tests pass.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [x] **3.1 Run full Rust & Python test suites:** `cargo test --workspace --locked` and `.venv/bin/pytest tests/native` passed.
- [x] **3.2 Run docs & security gates:** `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo deny check advisories licenses bans` passed.
- [x] **3.3 Conductor Review (Phase 3 & Track Sign-off):** All gates passed; marked complete in `conductor/tracks.md`.
